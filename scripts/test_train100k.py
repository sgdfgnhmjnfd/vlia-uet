#!/usr/bin/env python3
"""Tests for the 100k runs.

test_sampler_resume (sampler only, no model): with the training DataLoader settings (88,302 frames, batch 32, drop_last,
4 persistent workers, generator seeded 0),
  1. an uninterrupted ResumableRandomSampler gives the same batches as the 25k DataLoader(shuffle=True, generator=g);
  2. resuming at step 2,770 (just past the end of epoch 1: 2,759 steps per epoch) gives the same batches as running on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vlia_data.resumable_sampler import ResumableRandomSampler  # noqa: E402

N, B, SEED, WORKERS = 88302, 32, 0, 4


class Idx(Dataset):
    def __len__(self):
        return N

    def __getitem__(self, i):
        return i


def loader(sampler_kind: str):
    g = torch.Generator()
    g.manual_seed(SEED)
    ds = Idx()
    kw = dict(batch_size=B, num_workers=WORKERS, drop_last=True, persistent_workers=True, generator=g)
    if sampler_kind == "shuffle":
        return DataLoader(ds, shuffle=True, **kw), None
    s = ResumableRandomSampler(ds, g)
    return DataLoader(ds, sampler=s, **kw), s


def batches(dl, sampler, start_step: int, end_step: int):
    """Batches of steps start_step+1 .. end_step, as the training loop takes them (resume at start_step)."""
    if start_step > 0:
        sampler.fast_forward(start_step, len(dl), B)
    out, step = [], start_step
    while step < end_step:
        for b in dl:
            if step >= end_step:
                break
            out.append(b.tolist())
            step += 1
    return out


def test_sampler_resume():
    spe = N // B
    assert spe == 2759
    end = 2 * spe + 30  # cross two epoch boundaries
    ref, _ = loader("shuffle")
    full_ref = batches(ref, None, 0, end)
    dl, s = loader("resumable")
    full = batches(dl, s, 0, end)
    assert full == full_ref, "uninterrupted resumable order differs from DataLoader(shuffle=True)"
    for resume_at in (20, 2770, spe, 2 * spe + 5):
        dl, s = loader("resumable")
        part = batches(dl, s, resume_at, end)
        assert part == full[resume_at:], f"resume at step {resume_at}: order differs"
        print(f"resume at step {resume_at}: {len(part)} batches identical to the uninterrupted run")
    # the 25k behaviour (fresh DataLoader on resume) would not match
    ref2, _ = loader("shuffle")
    assert batches(ref2, None, 0, 10) != full[2770:2780]


# ----------------------------------------------------------------------
# final evaluation rows (CPU, no env, no model)
# ----------------------------------------------------------------------
import json  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
US = ROOT / "results/train_100k/unit_sentences.json"
TABLE = Path("/media/dhqg/d3/vlia_outputs/train_100k/eval_token_table.npz")
EP_VLIA = Path("/media/dhqg/d3/vlia_outputs/libero_intention_only_v1/eval/episodes")
EP_BASE = Path("/media/dhqg/d3/vlia_outputs/libero_goal_ablation_v1/episodes")


def _load():
    import libero_goal_ablation as G

    us = json.loads(US.read_text())
    original = {int(k): v for k, v in us["original"].items()}
    ph = json.loads((ROOT / "results/stage11_libero_goal/phrasings.json").read_text())
    e2e = json.loads((ROOT / "results/stage12_end_to_end/e2e_tokens.json").read_text())["tokens"]
    return G, us, original, ph, e2e


def test_pairs_byte_identical():
    G, us, original, ph, e2e = _load()
    n = 0
    for lrow, irow in us["pairs"].items():
        assert set(us["rows"][lrow]) == set(us["rows"][irow]), lrow
        for k, u in us["rows"][lrow].items():
            assert u["instruction"].encode() == us["rows"][irow][k]["token_text"].encode(), (lrow, k)
            assert u["sentence"] == us["rows"][irow][k]["sentence"], (lrow, k)
            n += 1
    print(f"pairs: {n} units, instruction of every L-* row == token sentence of its partner, byte for byte")


def test_reuse_only_when_true():
    _, us, *_ = _load()
    for r, rows in us["rows"].items():
        for k, u in rows.items():
            assert (u["reuse"] is not None) <= (u["sentence"] == u["true"]), (r, k)
            if r in ("L-E2E", "L-sup-p25", "L-gen-p25", "L-para", "L-goal"):
                assert u["reuse"] in (None, "correct") and (u["reuse"] == "correct") == (u["sentence"] == u["true"]), (r, k)
            if r in ("E2E-sup-p25", "E2E-gen-p25"):
                assert u["reuse"] in (None, "I-only") and (u["reuse"] == "I-only") == (u["sentence"] == u["true"]), (r, k)
            if r in ("I-only", "I-wrong", "I-para", "I-goal", "E2E", "correct", "empty"):
                assert u["reuse"] is None, (r, k)
    print("reuse: only where the sentence equals the true description")


def test_unit_counts():
    _, us, _, _, e2e = _load()
    assert len(us["units_300"]) == 300 and len(us["units_299"]) == 299
    assert us["units_299"] == [k for k in us["units_300"] if k in e2e] and set(us["units_299"]) == set(e2e)
    for r, rows in us["rows"].items():
        want = 299 if r in ("E2E", "E2E-sup-p25", "E2E-gen-p25", "L-E2E", "L-sup-p25", "L-gen-p25") else 300
        assert len(rows) == want, (r, len(rows))
    print("units: 300 / 299, the 299 list equals the Stage 12 unit list")


def test_harness_matches_unit_file_and_table():
    import numpy as np

    G, us, original, ph, e2e = _load()
    table = set(map(str, np.load(TABLE)["tasks"]))
    for c in us["order"]:
        for k, u in us["rows"][c].items():
            t, e = map(int, k.split(","))
            instr, z, skip = G.unit_inputs(c, t, e, original, ph, e2e, {}, us["rows"])
            if c in G.UNIT_VLIA or c in G.UNIT_BASE:
                assert (skip is not None) == (u["reuse"] is not None), (c, k)
                if skip:
                    continue
            assert skip is None, (c, k, skip)
            assert instr == u["instruction"], (c, k)
            assert z == u["token_text"], (c, k)
            if z is not None:
                assert z in table, (c, k, z)
    print("harness unit_inputs == unit_sentences.json for every row and unit; every token is in the eval table")


def test_old_rows_match_25k_episodes():
    G, us, original, ph, e2e = _load()
    sup = {"tokens": {}, "reuse": {}}
    n = {}
    for d, conds in ((EP_VLIA, ("I-only", "I-wrong", "I-para", "I-goal", "E2E", "E2E-sup-p25")), (EP_BASE, ("correct", "empty"))):
        for p in sorted(d.glob("*.json")):
            r = json.loads(p.read_text())
            c, t, e = r.get("condition"), r.get("task"), r.get("episode")
            if c not in conds or e is None or e >= 30:
                continue
            instr, z, skip = G.unit_inputs(c, t, e, original, ph, e2e, {}, us["rows"])
            if c == "E2E-sup-p25":
                u = us["rows"][c][f"{t},{e}"]
                assert r["z_int_task"] == u["sentence"], (c, t, e)
                n[c] = n.get(c, 0) + 1
                continue
            assert skip is None and instr == r["instruction"], (c, t, e, instr, r["instruction"])
            if c not in ("correct", "empty"):
                assert z == r["z_int_task"], (c, t, e, z, r["z_int_task"])
            n[c] = n.get(c, 0) + 1
    print("old rows reproduce the 25k episode JSONs (instruction and token text):", n)
    assert n.get("I-only", 0) == 300 and n.get("correct", 0) >= 300 and n.get("E2E", 0) == 299


def main() -> int:
    tests = [test_pairs_byte_identical, test_reuse_only_when_true, test_unit_counts,
             test_harness_matches_unit_file_and_table, test_old_rows_match_25k_episodes]
    if "--units-only" not in sys.argv:
        tests.insert(0, test_sampler_resume)
    for f in tests:
        f()
    print(f"test_train100k passed ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
