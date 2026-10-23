#!/usr/bin/env python3
"""Intention Adapter dimension tests (prereg v2, Stage 3).

For each intention dim d in {256, 768, 1152}: finite loss, non-zero adapter
gradient, prefix length +1, intention token right before the state token,
intention present in the KV cache at inference, flag-off output identical to
upstream SmolVLA (max abs diff <= 1e-6), checkpoint round-trip.

CPU, fp32, randomly initialised SmolVLA (load_vlm_weights=False) with the
scalar architecture fields of the 7-D LIBERO checkpoint config. Implementation
correctness only.

    PYTHONPATH=. python scripts/test_adapter_dims.py --out results/adapter_dims_v1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching  # noqa: E402

from policies.smolvla.configuration_smolvla import VLIASmolVLAConfig  # noqa: E402
from policies.smolvla.modeling_smolvla import VLIAFlowMatching  # noqa: E402

DIMS = (256, 768, 1152)
TOL = 1e-6
CKPT_CONFIG = Path(
    "/media/dhqg/d1/vlia_outputs/oracle_matched_LIBERO7D_FIXED/baseline/seed_0/checkpoint_025000/pretrained_model/config.json"
)
TESTS = (
    "finite_loss",
    "adapter_grad_nonzero",
    "prefix_len_plus_1",
    "intention_before_state",
    "in_kv_cache",
    "flag_off_identical",
    "checkpoint_roundtrip",
)


def make_config(d: int, use_intention: bool) -> VLIASmolVLAConfig:
    kwargs = {}
    if CKPT_CONFIG.exists():
        raw = json.loads(CKPT_CONFIG.read_text())
        fields = VLIASmolVLAConfig.__dataclass_fields__
        for key, value in raw.items():
            if key in fields and key != "type" and isinstance(value, (int, float, str, bool)) and key != "device":
                kwargs[key] = value
    kwargs.update(use_intention_token=use_intention, intention_dim=d, load_vlm_weights=False, device="cpu")
    return VLIASmolVLAConfig(**kwargs)


def inputs(cfg, d: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    b = 2
    images = [torch.rand(b, 3, 512, 512, generator=g) * 2 - 1]
    img_masks = [torch.ones(b, dtype=torch.bool)]
    lang_tokens = torch.randint(1, 1000, (b, 12), generator=g)
    lang_masks = torch.ones(b, 12, dtype=torch.bool)
    state = torch.randn(b, cfg.max_state_dim, generator=g)
    actions = torch.randn(b, cfg.chunk_size, cfg.max_action_dim, generator=g)
    noise = torch.randn(b, cfg.chunk_size, cfg.max_action_dim, generator=g)
    z_int = torch.randn(b, d, generator=g)
    return images, img_masks, lang_tokens, lang_masks, state, actions, noise, z_int


def capture_kv(model, call):
    """Run `call` and return the past_key_values produced by the prefix pass."""
    captured = {}
    original = model.vlm_with_expert.forward

    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        if kwargs.get("fill_kv_cache") and "kv" not in captured:
            captured["kv"] = out[1]
        return out

    model.vlm_with_expert.forward = wrapped
    try:
        result = call()
    finally:
        model.vlm_with_expert.forward = original
    return result, captured["kv"]


def kv_len(kv) -> int:
    first = kv[min(kv)] if isinstance(kv, dict) else kv[0]
    return int(first["key_states"].shape[1])


def run_dim(d: int) -> dict:
    res = {}
    detail = {}
    torch.manual_seed(0)
    cfg_on = make_config(d, True)
    model = VLIAFlowMatching(cfg_on).eval()
    images, img_masks, lang_tokens, lang_masks, state, actions, noise, z_int = inputs(cfg_on, d)

    # 1-2: loss + adapter gradient
    model.train()
    losses = model.forward(images, img_masks, lang_tokens, lang_masks, state, actions,
                           noise=noise, time=torch.full((2,), 0.5), z_int=z_int)
    loss = losses.mean()
    loss.backward()
    grad = model.intention_adapter.proj.weight.grad
    res["finite_loss"] = bool(torch.isfinite(loss).item())
    res["adapter_grad_nonzero"] = bool(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0)
    detail["loss"] = float(loss)
    detail["adapter_grad_norm"] = float(grad.norm()) if grad is not None else None
    model.zero_grad(set_to_none=True)
    model.eval()

    with torch.no_grad():
        base_embs, base_pad, base_att = super(VLIAFlowMatching, model).embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state)
        embs, pad, att = model.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state, z_int=z_int)
        expected = model.intention_adapter(z_int).to(embs.dtype)
    # 3-4: prefix structure
    res["prefix_len_plus_1"] = embs.shape[1] == base_embs.shape[1] + 1
    res["intention_before_state"] = bool(
        torch.allclose(embs[:, -1], base_embs[:, -1], atol=1e-6)
        and torch.allclose(embs[:, -2], expected[:, 0], atol=1e-6)
        and bool(pad[:, -2].all()) and int(att[0, -2]) == 1
        and torch.equal(embs[:, :-2], base_embs[:, :-1])
    )
    detail["prefix_len"] = {"base": int(base_embs.shape[1]), "vlia": int(embs.shape[1])}

    # 5: KV cache
    with torch.no_grad():
        out_on, kv_on = capture_kv(model, lambda: model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone(), z_int=z_int))
        _, kv_none = capture_kv(model, lambda: model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone(), z_int=None))
        out_z2, kv_z2 = capture_kv(model, lambda: model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone(), z_int=z_int * -1.0))
    first = min(kv_on) if isinstance(kv_on, dict) else 0
    keys_differ = not torch.allclose(kv_on[first]["key_states"], kv_z2[first]["key_states"])
    res["in_kv_cache"] = bool(kv_len(kv_on) == kv_len(kv_none) + 1 == embs.shape[1] and keys_differ
                              and not torch.allclose(out_on, out_z2))
    detail["kv_len"] = {"with_intention": kv_len(kv_on), "without": kv_len(kv_none)}

    # 6: flag off / z_int None == upstream SmolVLA with the same weights
    torch.manual_seed(0)
    cfg_off = make_config(d, False)
    upstream = VLAFlowMatching(cfg_off).eval()
    missing, unexpected = upstream.load_state_dict(model.state_dict(), strict=False)
    assert not missing and all(k.startswith("intention_adapter") for k in unexpected), (missing, unexpected)
    off = VLIAFlowMatching(cfg_off).eval()
    off.load_state_dict(model.state_dict())
    with torch.no_grad():
        ref = upstream.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone())
        out_off = off.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone(), z_int=z_int)
        out_none = model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise.clone(), z_int=None)
        ref_prefix = upstream.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)[0]
        off_prefix = off.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state, z_int=z_int)[0]
    diff_off = float((out_off - ref).abs().max())
    diff_none = float((out_none - ref).abs().max())
    diff_prefix = float((off_prefix - ref_prefix).abs().max()) if off_prefix.shape == ref_prefix.shape else float("inf")
    res["flag_off_identical"] = max(diff_off, diff_none, diff_prefix) <= TOL
    detail["flag_off_max_abs_diff"] = {"flag_off": diff_off, "z_int_none": diff_none, "prefix": diff_prefix}
    del upstream, off

    # 7: checkpoint round trip
    tmp_root = Path("/media/dhqg/d3/tmp")
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        path = Path(tmp) / f"vlia_d{d}.pt"
        torch.save(model.state_dict(), path)
        torch.manual_seed(123)  # different init, must be overwritten by the load
        fresh = VLIAFlowMatching(make_config(d, True)).eval()
        fresh.load_state_dict(torch.load(path, map_location="cpu"))
        with torch.no_grad():
            out_fresh = fresh.sample_actions(images, img_masks, lang_tokens, lang_masks, state,
                                             noise=noise.clone(), z_int=z_int)
        diff_rt = float((out_fresh - out_on).abs().max())
        same_w = torch.equal(fresh.intention_adapter.proj.weight, model.intention_adapter.proj.weight)
    res["checkpoint_roundtrip"] = bool(diff_rt <= TOL and same_w)
    detail["roundtrip_max_abs_diff"] = diff_rt
    return {"d": d, "results": res, "detail": detail}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "results/adapter_dims_v1")
    ap.add_argument("--dims", type=int, nargs="+", default=list(DIMS))
    args = ap.parse_args()
    torch.set_num_threads(8)
    rows = []
    for d in args.dims:
        t0 = time.time()
        r = run_dim(d)
        r["seconds"] = time.time() - t0
        rows.append(r)
        print(json.dumps(r), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps({
        "rows": rows, "tolerance": TOL, "device": "cpu", "dtype": "float32",
        "config_source": str(CKPT_CONFIG), "torch": torch.__version__,
        "git_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    }, indent=2))
    lines = ["# Intention Adapter dimension tests (prereg v2, Stage 3)", "",
             "Implementation correctness only (CPU, fp32, randomly initialised SmolVLA with the 7-D LIBERO checkpoint "
             "architecture fields). Not scientific evidence.", "",
             "| test | " + " | ".join(f"d={r['d']}" for r in rows) + " |",
             "|---|" + "---|" * len(rows)]
    for t in TESTS:
        lines.append(f"| {t} | " + " | ".join("PASS" if r["results"][t] else "FAIL" for r in rows) + " |")
    lines += ["", "## Details", ""]
    for r in rows:
        lines.append(f"- d={r['d']}: " + json.dumps(r["detail"]))
    all_pass = all(all(r["results"].values()) for r in rows)
    lines += ["", f"Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}", ""]
    (args.out / "summary.md").write_text("\n".join(lines))
    print("ALL PASS" if all_pass else "SOME FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
