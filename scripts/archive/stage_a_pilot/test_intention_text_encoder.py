import torch

from models.intention_encoder import (
    IntentionVisualEncoder,
    IntentionTextEncoder,
)


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

text_encoder.eval()


texts = [
    "draw",
    "hold the sketch paper",
    "positioning reference materials",
]


with torch.no_grad():
    features = text_encoder(texts)


print("=== INTENTION TEXT ENCODER ===")
print("texts:", len(texts))
print("shape:", features.shape)
print("dtype:", features.dtype)
print(
    "finite:",
    torch.isfinite(features).all().item(),
)

assert features.shape == (3, 960)
assert torch.isfinite(features).all()

print()
print("INTENTION TEXT ENCODER: PASS")