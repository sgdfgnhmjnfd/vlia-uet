import json
from pathlib import Path
import matplotlib.pyplot as plt

HISTORY = Path(
    "/media/dhqg/d1/vlia_outputs/oracle_matched_LIBERO7D_FIXED/"
    "baseline/seed_0/history.json"
)

OUT = HISTORY.parent / "plots"
OUT.mkdir(exist_ok=True)

with open(HISTORY, "r") as f:
    h = json.load(f)

step = [x["step"] for x in h]
loss = [x["loss"] for x in h]
loss100 = [x["loss_100"] for x in h]
lr = [x["lr"] for x in h]
time_h = [x["elapsed_s"] / 3600 for x in h]

# 1. Loss
plt.figure(figsize=(8, 5))
plt.plot(step, loss, alpha=0.25, linewidth=0.8, label="Batch loss")
plt.plot(step, loss100, linewidth=2, label="Moving average (100)")
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("SmolVLA Baseline Training Loss")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "loss_vs_step.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT / "loss_vs_step.pdf", bbox_inches="tight")
plt.close()

# 2. Learning rate
plt.figure(figsize=(8, 5))
plt.plot(step, lr, linewidth=2)
plt.xlabel("Training step")
plt.ylabel("Learning rate")
plt.title("Learning Rate Schedule")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUT / "lr_vs_step.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT / "lr_vs_step.pdf", bbox_inches="tight")
plt.close()

# 3. Loss vs elapsed time
plt.figure(figsize=(8, 5))
plt.plot(time_h, loss100, linewidth=2)
plt.xlabel("Elapsed time (hours)")
plt.ylabel("Loss (100-step moving average)")
plt.title("Training Convergence")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUT / "loss_vs_time.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT / "loss_vs_time.pdf", bbox_inches="tight")
plt.close()

print("Saved plots to:", OUT)
