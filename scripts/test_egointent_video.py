from datasets.egointent_adapter import (
    build_egointent_alignment_samples,
)
from datasets.egointent_video import (
    sample_video_frames,
)

from datasets.egointent_video import (
    sample_video_frames,
    frames_to_tensor,
)
LABEL = (
    "/media/dhqg/d1/datasets/egointent/"
    "indoor/art_studio/draw/step_label.json"
)


samples = build_egointent_alignment_samples(
    LABEL
)

sample = samples[1]

frames = sample_video_frames(
    sample.observation["video_path"],
    num_frames=8,
    resize=(224, 224),
)
tensor = frames_to_tensor(frames)

print()
print("tensor shape:", tensor.shape)
print("tensor dtype:", tensor.dtype)
print("tensor min:", tensor.min().item())
print("tensor max:", tensor.max().item())

assert tensor.shape == (8, 3, 224, 224)
assert tensor.dtype.is_floating_point
assert 0.0 <= tensor.min().item()
assert tensor.max().item() <= 1.0

print()
print("EGOINTENT TENSOR CONTRACT: PASS")

print("=== EGOINTENT VIDEO TEST ===")
print("sample_id:", sample.sample_id)
print(
    "video:",
    sample.observation["video_path"],
)
print("WHY:", sample.why)

print()
print("frames shape:", frames.shape)
print("dtype:", frames.dtype)
print("min:", frames.min())
print("max:", frames.max())

assert frames.ndim == 4
assert frames.shape[-1] == 3
assert frames.shape[1:3] == (224, 224)
assert frames.dtype == "uint8"
assert len(frames) > 0

print()
print("EGOINTENT VIDEO DECODE: PASS")