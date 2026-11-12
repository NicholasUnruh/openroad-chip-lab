"""MXFP8's gain over BF16 at each level of measurement: ASAP7 unit (the numerics study's
prediction), SKY130HD unit, SKY130HD tile (V7 against V6). Reads docs/journal/results.json
(tools/journal/results_json.py) and the tile layout metrics in results/sky130/.
usage: uv run --no-project --with matplotlib python tools/journal/sky130_check_plot.py"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
rows = json.loads((root / "docs/journal/results.json").read_text())
value = {(r["format"], r["metric"]): r["value"] for r in rows}


def gain(metric):
    return value[("BF16", metric)] / value[("MXFP8", metric)]


def peak_density(tile, macs):
    m = json.loads((root / f"results/sky130/{tile}_layout.json").read_text())
    return macs / (m["finish__design__instance__area"] * 1e-6)


# Area gain is area per MAC (units) or peak MACs per clock per mm², cells plus SRAM macro (tiles).
area = [gain("area_per_mac_um2"), gain("sky130_area_per_mac_um2"),
        peak_density("training_tile_v7", 64) / peak_density("training_tile_v6", 32), None]
energy = [gain("energy_per_mac_pj"), gain("sky130_energy_per_mac_pj"),
          gain("tile_pipelined_energy_per_mac_pj"), gain("tile_gemm32_energy_per_mac_pj")]
labels = ["ASAP7 unit\n(prediction)", "SKY130 unit", "SKY130 tile\npipelined GEMMs", "SKY130 tile\n32³ GEMM"]

fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=120)
width = 0.38
for i, (a, e) in enumerate(zip(area, energy)):
    if a is not None:
        ax.bar(i - width / 2, a, width, color="#4c78a8", label="area" if i == 0 else None)
        ax.text(i - width / 2, a + 0.04, f"{a:.2f}×", ha="center", fontsize=8)
    ax.bar(i + width / 2, e, width, color="#e45756", label="energy per MAC" if i == 0 else None)
    ax.text(i + width / 2, e + 0.04, f"{e:.2f}×", ha="center", fontsize=8)
ax.axhline(1, color="grey", lw=0.8)
ax.set_xticks(range(len(labels)), labels, fontsize=8)
ax.set_ylabel("MXFP8 gain over BF16 (×)")
ax.set_ylim(0, 2.6)
ax.set_title("MXFP8 over BF16: the unit gain survives the PDK change, not the tile", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
out = root / "docs/journal/img/sky130-check.png"
fig.savefig(out)
print(out)
