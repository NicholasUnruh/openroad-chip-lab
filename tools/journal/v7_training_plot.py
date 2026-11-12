"""Test-loss curves of the MXFP8 training run on the V7 RTL against the study's C++ datapath
and FP32 (results/v7/train_*.txt), mean over seeds, plus the 1% gate.
usage: uv run --no-project --with matplotlib python tools/journal/v7_training_plot.py"""

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
runs = {
    "FP32": "train_fp32.txt",
    "MXFP8, C++ study datapath": "train_mxfp8.txt",
    "MXFP8, V7 RTL tile": "train_mxfp8_v7_tile.txt",
}


def curve(path):
    points = defaultdict(list)
    for line in (root / "results/v7" / path).read_text().splitlines():
        kind, _, _, step, _, test_loss, _ = (line.split() + [""] * 7)[:7]
        if kind in ("curve", "final"):
            points[int(step)].append(float(test_loss))
    steps = sorted(points)
    return steps, [sum(points[s]) / len(points[s]) for s in steps]


fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
styles = ["-", "-", "--"]
for (label, path), style in zip(runs.items(), styles):
    steps, loss = curve(path)
    ax.plot(
        steps,
        loss,
        style,
        marker="o",
        markersize=3,
        label=f"{label} (final {loss[-1]:.4f})",
    )
fp32_final = curve(runs["FP32"])[1][-1]
ax.axhline(
    fp32_final * 1.01, color="grey", linewidth=0.8, linestyle=":", label="1% gate"
)
ax.set_xlabel("SGD step")
ax.set_ylabel("Test loss (mean of 3 seeds)")
ax.set_xlim(250, 3050)  # step 0 (loss about 2.3) is off the chart
ax.set_ylim(0.80, 0.98)
ax.set_title("64-64-10 MLP, every GEMM on the MXFP8 tile")
ax.legend(fontsize=8)
fig.tight_layout()
out = root / "docs/journal/img/v7-rtl-training.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out)
print(out)
