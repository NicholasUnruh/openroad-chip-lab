"""Training-loss penalty of every format recipe against FP32 (results/numerics/train_*.txt), mean of
three seeds with the seed range, plus the 1% gate.
usage: uv run --no-project --with matplotlib python tools/journal/format_accuracy_plot.py"""

import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
# Recipe file, label. MXFP4 with stochastic rounding diverges and is drawn clipped.
recipes = [
    ("bf16_dot8", "BF16 dot8"),
    ("mxfp8", "MXFP8 E4M3"),
    ("mxfp8_e5m2_bwd", "MXFP8 E4M3 fwd / E5M2 bwd"),
    ("mxfp6_e2m3", "MXFP6 E2M3"),
    ("fp8_tensor", "FP8 per-tensor E4M3 / E5M2"),
    ("mxfp6_e3m2", "MXFP6 E3M2"),
    ("nvfp4_sr", "NVFP4, SR gradients"),
    ("mxfp4", "MXFP4"),
    ("nvfp4", "NVFP4"),
    ("mxfp4_sr", "MXFP4, SR gradients"),
]
CLIP = 20.0


def finals(name):
    lines = (root / "results/numerics" / f"train_{name}.txt").read_text().splitlines()
    return [float(line.split()[5]) for line in lines if line.startswith("final")]


fp32 = statistics.mean(finals("fp32"))
fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=120)
for row, (name, label) in enumerate(recipes):
    penalties = [100 * (v / fp32 - 1) for v in finals(name)]
    mean = statistics.mean(penalties)
    shown = min(mean, CLIP)
    color = "#2a7f62" if mean <= 1 else "#b5452b"
    ax.barh(row, shown, color=color)
    ax.plot(
        [min(p, CLIP) for p in penalties], [row] * len(penalties), "k|", markersize=8
    )
    text = f"{mean:+.2f}%" if mean < CLIP else f"{mean:+.0f}% (diverges)"
    right = max(shown, *(min(p, CLIP) for p in penalties))
    ax.text(right + 0.4, row, text, va="center", fontsize=8)
ax.axvline(1.0, color="grey", linestyle=":", linewidth=1, label="1% gate")
ax.set_yticks(range(len(recipes)), [label for _, label in recipes], fontsize=8)
ax.invert_yaxis()
ax.set_xlim(-0.5, CLIP + 6)
ax.set_xlabel("Final test loss vs FP32, % (bar: mean of 3 seeds; ticks: seeds)")
ax.set_title("Training penalty per format, 64-64-10 MLP, 3,000 steps")
ax.legend(loc="center right", fontsize=8)
fig.tight_layout()
out = root / "docs/journal/img/format-accuracy.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print(out)
