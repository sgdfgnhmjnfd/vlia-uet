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
    StageAAlignmentModel,
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

intention_encoder = IntentionEncoderV0(
    visual_encoder=visual_encoder,
    text_encoder=text_encoder,
    intention_dim=256,
).to(device)

model = StageAAlignmentModel(
    intention_encoder=intention_encoder,
    text_encoder=text_encoder,
).to(device)

model.train()


print("=== INPUT ===")
print("sample:", sample.sample_id)
print("task:", sample.task)
print("history:", sample.history)
print("WHY:", sample.why)
print("frames:", len(frames))
print("device:", device)


output = model(
    frames=frames,
    task=sample.task,
    history=sample.history,
    why=sample.why,
)


loss = output["loss"]
z_align = output["z_align"]
z_why = output["z_why"]
cosine = output["cosine_similarity"]


print()
print("=== FORWARD ===")
print("z_align shape:", z_align.shape)
print("z_why shape:", z_why.shape)
print("loss:", loss.item())
print("cosine:", cosine.item())

assert z_align.shape == (1, 960)
assert z_why.shape == (1, 960)

assert torch.isfinite(z_align).all()
assert torch.isfinite(z_why).all()
assert torch.isfinite(loss)


# ---------------------------------------------------------
# Backward test
# ---------------------------------------------------------

loss.backward()


fusion_grad = 0.0

for param in intention_encoder.fusion.parameters():
    if param.grad is not None:
        fusion_grad += (
            param.grad.abs().sum().item()
        )


projection_grad = 0.0

for param in (
    intention_encoder
    .intention_projection
    .parameters()
):
    if param.grad is not None:
        projection_grad += (
            param.grad.abs().sum().item()
        )


backbone_grad_found = False

for param in visual_encoder.vision_model.parameters():
    if param.grad is not None:
        backbone_grad_found = True
        break

for param in visual_encoder.connector.parameters():
    if param.grad is not None:
        backbone_grad_found = True
        break

for param in text_encoder.text_model.parameters():
    if param.grad is not None:
        backbone_grad_found = True
        break


print()
print("=== GRADIENTS ===")
print("fusion grad:", fusion_grad)
print(
    "intention projection grad:",
    projection_grad,
)
print(
    "frozen backbone grad found:",
    backbone_grad_found,
)


assert fusion_grad > 0.0

# Stage-A loss operates on z_align [960].
# The downstream 256-D VLIA projection is not part
# of this alignment loss yet.
assert projection_grad == 0.0

assert backbone_grad_found is False


print()
print("STAGE-A ALIGNMENT GRAPH: PASS")