import dataclasses
import gc
import hashlib
import sys
from pathlib import Path

import torch
from lerobot.configs import PreTrainedConfig

PROJECT_ROOT = Path(
    "/home/dhqg/vlia-uet"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(
        str(PROJECT_ROOT)
    )


from lerobot.policies.smolvla.configuration_smolvla import (
    SmolVLAConfig,
)

from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
)

from policies.smolvla.configuration_smolvla import (
    VLIASmolVLAConfig,
)

from policies.smolvla.modeling_smolvla import (
    VLIASmolVLAPolicy,
)


PRETRAINED = "lerobot/smolvla_base"


def tensor_digest(
    tensor: torch.Tensor,
) -> str:
    """
    Compute an exact byte-level SHA256 digest without
    depending on NumPy support for the tensor dtype.
    """

    tensor = (
        tensor
        .detach()
        .cpu()
        .contiguous()
    )

    raw = (
        tensor
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )

    return hashlib.sha256(
        raw
    ).hexdigest()


def state_digests(
    model,
):
    result = {}

    for key, value in (
        model.state_dict().items()
    ):
        result[key] = {
            "shape": tuple(
                value.shape
            ),
            "dtype": str(
                value.dtype
            ),
            "digest": tensor_digest(
                value
            ),
        }

    return result


def make_vlia_config(
    base_cfg,
):
    """
    Clone every initialized SmolVLA dataclass field
    into VLIASmolVLAConfig, then add VLIA fields.
    """

    kwargs = {}

    for field in dataclasses.fields(
        SmolVLAConfig
    ):
        if not field.init:
            continue

        kwargs[field.name] = getattr(
            base_cfg,
            field.name,
        )

    return VLIASmolVLAConfig(
        **kwargs,
        use_intention_token=True,
        intention_dim=960,
    )


def main():
    print("=" * 72)
    print("MATCHED PRETRAINED INITIALIZATION TEST")
    print("=" * 72)

    print(
        "checkpoint:",
        PRETRAINED,
    )

    print()
    print("=== LOAD BASE CONFIG ===")

    # Keep the test on CPU so baseline and VLIA do not
    # simultaneously consume GPU memory.
    base_cfg = (
        PreTrainedConfig.from_pretrained(
            pretrained_name_or_path=PRETRAINED,
            device="cpu",
            num_vlm_layers=16,
            resize_imgs_with_padding=(
                224,
                224,
            ),
        )
    )

    if not isinstance(
        base_cfg,
        SmolVLAConfig,
    ):
        raise TypeError(
            "Expected SmolVLAConfig, got "
            f"{type(base_cfg).__name__}"
        )

    print(
        "config type:",
        type(base_cfg).__name__,
    )

    print(
        "device:",
        base_cfg.device,
    )

    print(
        "num_vlm_layers:",
        base_cfg.num_vlm_layers,
    )

    print(
        "chunk_size:",
        base_cfg.chunk_size,
    )

    print(
        "n_action_steps:",
        base_cfg.n_action_steps,
    )

    print(
        "resize:",
        base_cfg.resize_imgs_with_padding,
    )

    assert (
        base_cfg.chunk_size
        == 50
    )

    assert (
        base_cfg.n_action_steps
        == 50
    )

    assert (
        base_cfg.num_vlm_layers
        == 16
    )

    print()
    print("=== BASELINE LOAD ===")

    baseline = (
        SmolVLAPolicy.from_pretrained(
            PRETRAINED,
            config=base_cfg,
            strict=False,
        )
    )

    baseline.eval()

    baseline_digests = (
        state_digests(
            baseline
        )
    )

    print(
        "baseline state tensors:",
        len(
            baseline_digests
        ),
    )

    baseline_num_params = sum(
        p.numel()
        for p in baseline.parameters()
    )

    print(
        "baseline parameters:",
        baseline_num_params,
    )

    # Release baseline before constructing the VLIA model.
    del baseline

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print()
    print("=== BUILD VLIA CONFIG ===")

    vlia_cfg = make_vlia_config(
        base_cfg
    )

    print(
        "config type:",
        type(vlia_cfg).__name__,
    )

    print(
        "use_intention_token:",
        vlia_cfg.use_intention_token,
    )

    print(
        "intention_dim:",
        vlia_cfg.intention_dim,
    )

    print(
        "num_vlm_layers:",
        vlia_cfg.num_vlm_layers,
    )

    print(
        "chunk_size:",
        vlia_cfg.chunk_size,
    )

    print(
        "resize:",
        vlia_cfg.resize_imgs_with_padding,
    )

    assert (
        vlia_cfg.intention_dim
        == 960
    )

    assert (
        vlia_cfg.use_intention_token
        is True
    )

    print()
    print("=== VLIA LOAD ===")

    vlia = (
        VLIASmolVLAPolicy.from_pretrained(
            PRETRAINED,
            config=vlia_cfg,
            strict=False,
        )
    )

    vlia.eval()

    vlia_digests = (
        state_digests(
            vlia
        )
    )

    print(
        "VLIA state tensors:",
        len(
            vlia_digests
        ),
    )

    vlia_num_params = sum(
        p.numel()
        for p in vlia.parameters()
    )

    print(
        "VLIA parameters:",
        vlia_num_params,
    )

    print(
        "extra parameters:",
        vlia_num_params
        - baseline_num_params,
    )

    print()
    print("=== STATE-DICT STRUCTURE ===")

    baseline_keys = set(
        baseline_digests
    )

    vlia_keys = set(
        vlia_digests
    )

    missing_from_vlia = sorted(
        baseline_keys
        - vlia_keys
    )

    extra_in_vlia = sorted(
        vlia_keys
        - baseline_keys
    )

    print(
        "baseline keys missing in VLIA:",
        len(
            missing_from_vlia
        ),
    )

    for key in (
        missing_from_vlia[:20]
    ):
        print(
            "  MISSING:",
            key,
        )

    print(
        "extra VLIA keys:",
        len(
            extra_in_vlia
        ),
    )

    for key in extra_in_vlia:
        print(
            "  EXTRA:",
            key,
            tuple(
                vlia.state_dict()[
                    key
                ].shape
            ),
        )

    if missing_from_vlia:
        raise RuntimeError(
            "VLIA is missing baseline "
            "state-dict tensors"
        )

    non_adapter_extra = [
        key
        for key in extra_in_vlia
        if not key.startswith(
            "model.intention_adapter."
        )
    ]

    if non_adapter_extra:
        raise RuntimeError(
            "Unexpected non-adapter VLIA "
            f"parameters: {non_adapter_extra}"
        )

    if not extra_in_vlia:
        raise RuntimeError(
            "VLIA has no additional "
            "intention adapter parameters"
        )

    print()
    print(
        "=== EXACT SHARED-WEIGHT CHECK ==="
    )

    mismatched = []

    for key in sorted(
        baseline_keys
    ):
        baseline_info = (
            baseline_digests[
                key
            ]
        )

        vlia_info = (
            vlia_digests[
                key
            ]
        )

        if (
            baseline_info[
                "shape"
            ]
            != vlia_info[
                "shape"
            ]
            or baseline_info[
                "dtype"
            ]
            != vlia_info[
                "dtype"
            ]
            or baseline_info[
                "digest"
            ]
            != vlia_info[
                "digest"
            ]
        ):
            mismatched.append(
                key
            )

    print(
        "shared tensors checked:",
        len(
            baseline_keys
        ),
    )

    print(
        "mismatched shared tensors:",
        len(
            mismatched
        ),
    )

    for key in mismatched[:20]:
        print(
            "  MISMATCH:",
            key,
        )

    if mismatched:
        raise RuntimeError(
            "Baseline and VLIA do not "
            "share identical initialization"
        )

    print()
    print(
        "=== INTENTION ADAPTER ==="
    )

    adapter = (
        vlia
        .model
        .intention_adapter
    )

    print(
        "adapter class:",
        type(
            adapter
        ).__name__,
    )

    adapter_params = sum(
        p.numel()
        for p in adapter.parameters()
    )

    print(
        "adapter parameters:",
        adapter_params,
    )

    for name, param in (
        adapter.named_parameters()
    ):
        print(
            f"  {name}: "
            f"{tuple(param.shape)} "
            f"{param.dtype}"
        )

    # Direct adapter smoke test.
    z = torch.randn(
        2,
        960,
        device=next(
            adapter.parameters()
        ).device,
        dtype=next(
            adapter.parameters()
        ).dtype,
    )

    with torch.no_grad():
        projected = adapter(
            z
        )

    print(
        "adapter input:",
        tuple(z.shape),
    )

    print(
        "adapter output:",
        tuple(
            projected.shape
        ),
    )

    assert projected.shape == (
        2,
        1,
        960,
    )

    assert torch.isfinite(
        projected
    ).all()

    print()
    print("=" * 72)
    print(
        "COMMON PRETRAINED INITIALIZATION: PASS"
    )
    print("=" * 72)

    print(
        "Baseline and Oracle VLIA share "
        "identical pretrained SmolVLA weights."
    )

    print(
        "The only additional VLIA parameters "
        "are the intention adapter."
    )


if __name__ == "__main__":
    main()