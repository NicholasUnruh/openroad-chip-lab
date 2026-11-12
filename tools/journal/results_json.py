"""Write docs/journal/results.json: per-format area and energy per MAC (ASAP7, routed) and the
training loss penalty against FP32, from the same data as tools/numerics/report.py.

usage: python3 tools/journal/results_json.py
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "numerics"))
import report  # noqa: E402

OUT = report.ROOT / "docs/journal/results.json"
SOURCE = "docs/numerics-results.md"
# Format name, PPA harness (None: same hardware as another row), MACs per clock, training recipe.
FORMATS = [
    ("BF16", "ppa_dot8_bf16", 8, "bf16_dot8"),
    ("FP8 per-tensor", "ppa_dot8_e4m3", 8, "fp8_tensor"),
    ("MXFP8", "ppa_dot32_mxfp8", 32, "mxfp8"),
    # The E5M2 unit's recipe keeps E4M3 forward and uses E5M2 for the gradients.
    ("MXFP8 E5M2", "ppa_dot32_mxfp8_e5m2", 32, "mxfp8_e5m2_bwd"),
    ("MXFP6 E2M3", "ppa_dot32_mxfp6_e2m3", 32, "mxfp6_e2m3"),
    ("MXFP6 E3M2", "ppa_dot32_mxfp6_e3m2", 32, "mxfp6_e3m2"),
    ("MXFP4", "ppa_dot32_mxfp4", 32, "mxfp4"),
    ("MXFP4-SR", None, 32, "mxfp4_sr"),
    ("NVFP4", "ppa_dot32_nvfp4", 32, "nvfp4"),
    ("NVFP4-SR", None, 32, "nvfp4_sr"),
]
# A run counts as diverged, with no penalty value, when its mean final test loss exceeds this
# multiple of FP32's (the worst converging recipe is 1.17x).
DIVERGED = 2.0


def main(argv):
    if argv[1:]:
        print(__doc__)
        return 0 if argv[1:] == ["--help"] else 2
    train = report.training()
    fp32 = train["fp32"][0]
    rows = []
    for name, harness, macs, recipe in FORMATS:
        if harness:
            p = report.ppa(harness, macs)
            rows.append({"format": name, "metric": "area_per_mac_um2",
                         "value": round(p["area_per_mac"], 1), "unit": "um2 (ASAP7)", "source": SOURCE})
            rows.append({"format": name, "metric": "energy_per_mac_pj",
                         "value": round(p["fj_per_mac"] / 1e3, 3), "unit": "pJ (ASAP7)", "source": SOURCE})
        loss = train[recipe][0]
        row = {"format": name, "metric": "loss_penalty_pct", "value": None,
               "unit": "% test loss vs FP32", "source": SOURCE}
        if math.isfinite(loss) and loss <= DIVERGED * fp32:
            row["value"] = round((loss / fp32 - 1) * 100, 2)
        else:
            row["note"] = f"diverged (mean test loss {loss:.2f} vs FP32 {fp32:.4f})"
        rows.append(row)
    OUT.write_text("[\n" + ",\n".join("  " + json.dumps(r, ensure_ascii=False) for r in rows) + "\n]\n")
    print(f"wrote {len(rows)} rows to {OUT.relative_to(report.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
