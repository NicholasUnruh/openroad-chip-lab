"""Silicon cost of each routed dot-product unit (ASAP7 area and energy per MAC), coloured by
whether its format trains within the 1% gate (docs/journal/results.json, from
tools/journal/results_json.py).
usage: uv run --no-project --with matplotlib python tools/journal/format_cost_plot.py"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
rows = json.loads((root / "docs/journal/results.json").read_text())
value = {(r["format"], r["metric"]): r["value"] for r in rows}
# Each unit is scored with its best recipe: NVFP4 trains better with stochastic rounding.
best = {"NVFP4": "NVFP4-SR"}
# Label offsets in points, placed by hand so no two labels overlap.
offsets = {
    "BF16": (-8, 0), "FP8 per-tensor": (-8, 0), "MXFP8 E5M2": (8, 4), "MXFP8": (-8, 0),
    "MXFP6 E2M3": (40, -28), "MXFP6 E3M2": (40, 22), "NVFP4": (-30, 22), "MXFP4": (8, 0),
}

fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=120)
for f in offsets:
    penalty = value[(best.get(f, f), "loss_penalty_pct")]
    passes = penalty <= 1
    x, y = value[(f, "energy_per_mac_pj")], value[(f, "area_per_mac_um2")]
    # Passing units are filled and drawn on top; failing ones are rings.
    if passes:
        ax.scatter(x, y, s=60, color="#2a7f62", zorder=4)
    else:
        ax.scatter(x, y, s=90, facecolors="none", edgecolors="#b5452b", linewidths=1.8, zorder=3)
    dx, dy = offsets[f]
    sr = " SR" if f in best else ""
    leader = {"arrowstyle": "-", "color": "grey", "lw": 0.6} if abs(dy) > 10 else None
    ax.annotate(f"{f}  {penalty:+.2f}%{sr}", (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=8, va="center", ha="right" if dx < 0 else "left", arrowprops=leader)
ax.set_xlim(0, 1.1)
ax.set_ylim(0, 185)
ax.set_xlabel("Energy per MAC, pJ (simulated activity, 1 GHz)")
ax.set_ylabel("Standard-cell area per MAC, µm²")
ax.set_title("Routed ASAP7 dot-product units: filled trains within 1% of FP32, ring fails",
             fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
out = root / "docs/journal/img/format-cost.png"
fig.savefig(out)
print(out)
