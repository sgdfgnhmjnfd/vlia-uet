import torch

from datasets.egointent_adapter import (
    build_egointent_alignment_samples,
)
from datasets.egointent_video import (
    sample_video_frames,
)
from models.intention_encoder import (
    IntentionVisualEncoder,
)


LABEL = (
    "/media/dhqg/d1/datasets/egointent/"
    "indoor/art_studio/draw/step_label.json"
)


samples = build_egointent_alignment_samples(
    LABEL
)

sample = samples[1]

# IMPORTANT:
# Do not manually resize or normalize here.
# SmolVLM's official image processor handles preprocessing.
frames = sample_video_frames(
    sample.observation["video_path"],
    num_frames=8,
    resize=None,
)

frames = list(frames)


print("=== INPUT ===")
print("sample:", sample.sample_id)
print("frames:", len(frames))
print("raw shape:", frames[0].shape)
print("WHY:", sample.why)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("device:", device)


encoder = IntentionVisualEncoder(
    intention_dim=256,
).to(device)

encoder.eval()


# ---------------------------------------------------------
# Test official SmolVLM preprocessing and tile grouping.
# ---------------------------------------------------------

pixel_values, tiles_per_frame = (
    encoder.preprocess_frames(frames)
)

print()
print("=== PREPROCESSOR ===")
print(
    "pixel_values shape:",
    pixel_values.shape,
)
print(
    "pixel_values dtype:",
    pixel_values.dtype,
)
print(
    "tiles per frame:",
    tiles_per_frame,
)
print(
    "total tiles:",
    sum(tiles_per_frame),
)

assert len(tiles_per_frame) == len(frames)

assert (
    sum(tiles_per_frame)
    == pixel_values.shape[0]
)


# ---------------------------------------------------------
# Test frozen 960-D visual feature.
# ---------------------------------------------------------

with torch.no_grad():
    video_feature = (
        encoder.encode_video_features(
            frames
        )
    )

print()
print("=== FROZEN VISUAL FEATURE ===")
print(
    "shape:",
    video_feature.shape,
)
print(
    "dtype:",
    video_feature.dtype,
)
print(
    "finite:",
    torch.isfinite(
        video_feature
    ).all().item(),
)

assert video_feature.shape == (
    1,
    960,
)

assert torch.isfinite(
    video_feature
).all()


# ---------------------------------------------------------
# Test projected 256-D visual representation.
# ---------------------------------------------------------

with torch.no_grad():
    z_visual = encoder(
        frames
    )

print()
print("=== OUTPUT ===")
print(
    "z_visual shape:",
    z_visual.shape,
)
print(
    "dtype:",
    z_visual.dtype,
)
print(
    "finite:",
    torch.isfinite(
        z_visual
    ).all().item(),
)

assert z_visual.shape == (
    1,
    256,
)

assert torch.isfinite(
    z_visual
).all()


print()
print(
    "INTENTION VISUAL ENCODER: PASS"
)