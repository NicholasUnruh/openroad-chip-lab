"""Print the numerics-study tables (docs/numerics-results.md) from routed metrics and accuracy runs.

Reads build/ppa/<harness>/{metrics.json,power.txt} (make ppa-all) and build/numerics/
{gemm_error.txt,train_*.txt} (make numerics). ASAP7 figures are predictive: compare rows with each
other, never with foundry data.

usage: python3 tools/numerics/report.py
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Harness, MACs per clock, training recipes that exercise its arithmetic (empty: inference only).
# A point trains when any of its recipes passes.
POINTS = [
    ("ppa_fma_bf16", 1, ("bf16_fma",)),
    ("ppa_dot8_bf16", 8, ("bf16_dot8",)),
    ("ppa_dot8_bf16_f24", 8, ("bf16_dot8_f24",)),
    ("ppa_dot8_bf16_f28", 8, ("bf16_dot8_f28",)),
    ("ppa_dot8_bf16_f36", 8, ("bf16_dot8_f36",)),
    ("ppa_dot8_bf16_f40", 8, ("bf16_dot8_f40",)),
    ("ppa_dot8_e4m3", 8, ("fp8_tensor",)),
    ("ppa_dot8_e5m2", 8, ("fp8_tensor",)),
    ("ppa_dot32_mxfp8", 32, ("mxfp8",)),
    ("ppa_dot32_mxfp8_e5m2", 32, ("mxfp8_e5m2_bwd",)),
    ("ppa_dot32_mxfp6_e3m2", 32, ("mxfp6_e3m2",)),
    ("ppa_dot32_mxfp6_e2m3", 32, ("mxfp6_e2m3",)),
    ("ppa_dot32_mxfp4", 32, ("mxfp4", "mxfp4_sr")),
    ("ppa_dot32_nvfp4", 32, ("nvfp4", "nvfp4_sr")),
    ("ppa_dot8_int8", 8, ()),
]
# Training passes when its mean final test loss is within this fraction of FP32's.
LOSS_TOLERANCE = 0.01


def ppa(harness, macs):
    directory = ROOT / "build/ppa" / harness
    m = json.loads((directory / "metrics.json").read_text())
    total = None
    for line in (directory / "power.txt").read_text().splitlines():
        if line.startswith("Total"):
            total = float(line.split()[4])
    area = m["finish__design__instance__area__stdcell"]
    fmax = m["finish__timing__fmax"] / 1e6
    return {
        "area": area,
        "area_per_mac": area / macs,
        "fmax": fmax,
        "slack": m["finish__timing__setup__ws"],
        "power_mw": total * 1e3,
        "fj_per_mac": total / (1e9 * macs) * 1e15,
        # GMAC/s per mm^2 at the routed fmax; MACs per nJ at 1 GHz activity.
        "gmacs_mm2": macs * fmax / 1e3 / (area * 1e-6),
        "macs_nj": 1e6 / (total / (1e9 * macs) * 1e15),
    }


def training():
    runs = {}
    for path in sorted((ROOT / "build/numerics").glob("train_*.txt")):
        for line in path.read_text().splitlines():
            f = line.split()
            if f[0] == "final":
                runs.setdefault(f[1], []).append((float(f[5]), float(f[6])))
    out = {}
    for recipe, values in runs.items():
        losses = [v[0] for v in values]
        out[recipe] = (
            statistics.mean(losses),
            statistics.pstdev(losses),
            statistics.mean(v[1] for v in values),
            len(values),
        )
    return out


def gemm_errors():
    table = {}
    for line in (ROOT / "build/numerics/gemm_error.txt").read_text().splitlines()[1:]:
        dist, k, path, median, p99 = line.split()
        table[(dist, int(k), path)] = (float(median), float(p99))
    return table


def main(argv):
    if argv[1:]:
        print(__doc__)
        return 0 if argv[1:] == ["--help"] else 2
    train = training()
    fp32 = train["fp32"][0]
    passes = {r: v[0] <= fp32 * (1 + LOSS_TOLERANCE) for r, v in train.items()}

    print("## PPA (ASAP7, routed, 1 GHz target)\n")
    print(
        "| Point | MACs/clk | Area µm² | Area/MAC µm² | fmax MHz | Power at 1 GHz mW | Energy fJ/MAC |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    rows = {}
    for harness, macs, _ in POINTS:
        p = rows[harness] = ppa(harness, macs)
        print(
            f"| `{harness[4:]}` | {macs} | {p['area']:.1f} | {p['area_per_mac']:.1f} | {p['fmax']:.0f} | "
            f"{p['power_mw']:.2f} | {p['fj_per_mac']:.0f} |"
        )

    print("\n## GEMM error (median / p99 of |y - y64| / sum|a b|)\n")
    errors = gemm_errors()
    paths = sorted(
        {k[2] for k in errors}, key=lambda p: [k[2] for k in errors].index(p)
    )
    columns = [
        (d, k) for d in ("gaussian", "laplace", "outliers", "rows") for k in (256, 4096)
    ]
    print("| Datapath | " + " | ".join(f"{d} K={k}" for d, k in columns) + " |")
    print("| --- |" + " ---: |" * len(columns))
    for path in paths:
        cells = [
            f"{errors[(d, k, path)][0]:.2e} / {errors[(d, k, path)][1]:.2e}"
            for d, k in columns
        ]
        print(f"| `{path}` | " + " | ".join(cells) + " |")

    print(
        f"\n## Training emulation (final test loss, tolerance {LOSS_TOLERANCE:.0%} of FP32 = "
        f"{fp32 * (1 + LOSS_TOLERANCE):.4f})\n"
    )
    print(
        "| Recipe | Seeds | Test loss mean | Seed std | Test accuracy | Within tolerance |"
    )
    print("| --- | ---: | ---: | ---: | ---: | --- |")
    for recipe, (loss, std, acc, n) in train.items():
        print(
            f"| `{recipe}` | {n} | {loss:.4f} | {std:.4f} | {acc:.3f} | {'yes' if passes[recipe] else 'no'} |"
        )

    print("\n## Ranking (training-usable points; relative to `dot8_bf16`)\n")
    base = rows["ppa_dot8_bf16"]
    ranked = []
    for harness, _, recipes in POINTS:
        p = rows[harness]
        usable = any(passes.get(r, False) for r in recipes)
        ranked.append((usable, p["gmacs_mm2"], harness, recipes, p))
    ranked.sort(key=lambda r: (not r[0], -r[1]))
    print("| Rank | Point | Trains | GMAC/s/mm² | × dot8_bf16 | MAC/nJ | × dot8_bf16 |")
    print("| ---: | --- | --- | ---: | ---: | ---: | ---: |")
    for rank, (usable, _, harness, recipes, p) in enumerate(ranked, 1):
        trains = "inference only" if not recipes else ("yes" if usable else "no")
        print(
            f"| {rank if usable else '—'} | `{harness[4:]}` | {trains} | {p['gmacs_mm2']:.0f} | "
            f"{p['gmacs_mm2'] / base['gmacs_mm2']:.2f} | {p['macs_nj']:.0f} | {p['macs_nj'] / base['macs_nj']:.2f} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
