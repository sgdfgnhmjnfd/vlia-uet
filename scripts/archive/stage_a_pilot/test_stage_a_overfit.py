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


torch.manual_seed(0)


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


optimizer = torch.optim.AdamW(
    intention_encoder.fusion.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)


print("=== STAGE-A OVERFIT SMOKE TEST ===")
print("sample:", sample.sample_id)
print("task:", sample.task)
print("history:", sample.history)
print("WHY:", sample.why)
print("device:", device)


# ---------------------------------------------------------
# Cache frozen features.
#
# This avoids repeatedly running the expensive frozen
# SmolVLM visual/text backbone during the smoke test.
# ---------------------------------------------------------

with torch.no_grad():
    visual_feature = (
        visual_encoder.encode_video_features(
            frames
        )
    )

    task_feature = text_encoder(
        [sample.task]
    )

    if len(sample.history) == 0:
        history_feature = torch.zeros(
            1,
            960,
            dtype=torch.float32,
            device=device,
        )
    else:
        history_text = " ; ".join(
            sample.history
        )

        history_feature = text_encoder(
            [history_text]
        )

    z_why = text_encoder(
        [sample.why]
    ).detach()


fused_feature = torch.cat(
    [
        visual_feature,
        task_feature,
        history_feature,
    ],
    dim=-1,
).detach()


def compute_loss():
    z_align = intention_encoder.fusion(
        fused_feature
    )

    cosine = torch.nn.functional.cosine_similarity(
        z_align,
        z_why,
        dim=-1,
    )

    loss = 1.0 - cosine.mean()

    return loss, cosine


with torch.no_grad():
    initial_loss, initial_cosine = (
        compute_loss()
    )


print()
print(
    "initial loss:",
    initial_loss.item(),
)
print(
    "initial cosine:",
    initial_cosine.item(),
)


num_steps = 20


for step in range(1, num_steps + 1):
    optimizer.zero_grad()

    loss, cosine = compute_loss()

    loss.backward()

    optimizer.step()

    if (
        step == 1
        or step % 5 == 0
        or step == num_steps
    ):
        print(
            f"step {step:02d} | "
            f"loss={loss.item():.6f} | "
            f"cosine={cosine.item():.6f}"
        )


with torch.no_grad():
    final_loss, final_cosine = (
        compute_loss()
    )


print()
print("=== RESULT ===")
print(
    "initial loss:",
    initial_loss.item(),
)
print(
    "final loss:",
    final_loss.item(),
)
print(
    "initial cosine:",
    initial_cosine.item(),
)
print(
    "final cosine:",
    final_cosine.item(),
)


assert torch.isfinite(final_loss)

assert (
    final_loss.item()
    < initial_loss.item()
)

assert (
    final_cosine.item()
    > initial_cosine.item()
)


print()
print(
    "STAGE-A OVERFIT SMOKE TEST: PASS"
)