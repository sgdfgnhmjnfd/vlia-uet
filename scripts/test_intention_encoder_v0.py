import torch

from datasets.egointent_adapter import (
    build_egointent_alignment_samples,
)
from datasets.egointent_video import (
    sample_video_frames,
)
from models.intention_encoder import (
    IntentionEncoderV0,
    IntentionTextEncoder,
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


frames = sample_video_frames(
    sample.observation["video_path"],
    num_frames=8,
    resize=None,
)

frames = list(frames)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


visual_encoder = IntentionVisualEncoder(
    intention_dim=256,
).to(device)

text_encoder = IntentionTextEncoder(
    vlm=visual_encoder.vlm,
    processor=visual_encoder.processor,
).to(device)

encoder = IntentionEncoderV0(
    visual_encoder=visual_encoder,
    text_encoder=text_encoder,
    intention_dim=256,
).to(device)

encoder.eval()


print("=== INPUT ===")
print("sample:", sample.sample_id)
print("task:", sample.task)
print("history:", sample.history)
print("WHY:", sample.why)
print("frames:", len(frames))
print("device:", device)


with torch.no_grad():
    z_int = encoder(
        frames=frames,
        task=sample.task,
        history=sample.history,
    )


print()
print("=== OUTPUT ===")
print("z_int shape:", z_int.shape)
print("dtype:", z_int.dtype)
print(
    "finite:",
    torch.isfinite(z_int).all().item(),
)


assert z_int.shape == (1, 256)
assert torch.isfinite(z_int).all()


# Frozen backbone must remain frozen.
assert not any(
    p.requires_grad
    for p in visual_encoder.vision_model.parameters()
)

assert not any(
    p.requires_grad
    for p in visual_encoder.connector.parameters()
)

assert not any(
    p.requires_grad
    for p in text_encoder.text_model.parameters()
)


# Fusion head must remain trainable.
assert any(
    p.requires_grad
    for p in encoder.fusion.parameters()
)


print()
print("INTENTION ENCODER V0: PASS")