# Plot the cosine scheduling rates for the encoder and decoder

from src.pcd_config import PCDConfig
from src.scripts.training.lr_scheduling import _step_cosine_schedule_with_warmup
import matplotlib.pyplot as plt


config = PCDConfig()
total = config.max_train_steps
warmup = config.warmup_steps

if warmup == 0: 
    print("No warmup being used")

steps = list(range(total))
cos_schedule_fn = _step_cosine_schedule_with_warmup(
    warmup_steps=warmup,
    total_steps=total,
)

lr_multipliers = [cos_schedule_fn(s) for s in steps]

lr_schedule = [config.lr * multiplier for multiplier in lr_multipliers]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(steps, lr_schedule, label=f"encoder (peak={config.lr:.0e})")

ax.axvline(warmup, color="gray", linestyle="--", alpha=0.6, label=f"warmup end ({warmup})")
ax.set_xlabel("step")
ax.set_ylabel("learning rate")
ax.set_title(f"Cosine schedule w/ warmup — {warmup} warmup / {total} total")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

out_path = "out/lr_schedule.png"
fig.savefig(out_path, dpi=120)
print(f"Saved {out_path}")