"""Figures for the training-* journal entries (docs/journal/img/training-*.png).

usage: uv run --extra cpu python figures.py formats
       uv run --extra cpu python figures.py curves --runs <dir> [--out name]
       uv run --extra cpu python figures.py throughput --log <file>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

import mx  # noqa: E402

IMG = Path(__file__).resolve().parents[3] / "docs" / "journal" / "img"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# One fixed colour per format in every figure (categorical slots in fixed order).
COLOR = {
    "bf16": "#52514e",
    "mxfp8": "#2a78d6",
    "mxfp6_e3m2": "#eb6834",
    "mxfp6_e2m3": "#1baf7a",
    "mxfp4": "#eda100",
    "nvfp4": "#e87ba4",
}
LABEL = {
    "bf16": "BF16",
    "mxfp8": "MXFP8 E4M3",
    "mxfp6_e3m2": "MXFP6 E3M2",
    "mxfp6_e2m3": "MXFP6 E2M3",
    "mxfp4": "MXFP4 E2M1",
    "nvfp4": "NVFP4",
}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def formats() -> None:
    """Element grids and per-element quantization error on Gaussian data."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6.2), height_ratios=[1, 1.4])
    elements = [("mxfp8", mx.E4M3), ("mxfp6_e3m2", mx.E3M2), ("mxfp6_e2m3", mx.E2M3),
                ("mxfp4", mx.E2M1)]  # fmt: skip
    for row, (name, e) in enumerate(elements):
        top_value = int(e.max_value * 2.0**-e.grid)
        grid = torch.arange(1, top_value + 1, dtype=torch.float64)
        values = mx.dequantize(mx.quantize(grid * 2.0**e.grid, mx.Format("g", e, 1)))
        values = torch.unique(values[values > 0])
        top.scatter(values, [row] * len(values), s=10, color=COLOR[name], linewidths=0)
        top.text(e.max_value * 1.4, row, f"{len(values)} positive values", va="center",
                 fontsize=8, color=MUTED)  # fmt: skip
    top.set_xscale("log", base=2)
    top.set_yticks(range(len(elements)), [LABEL[n] for n, _ in elements], fontsize=9)
    top.set_xlim(2.0**-10, 2.0**12)
    top.set_title("Representable positive element values (before the block scale)",
                  fontsize=10, color=INK, loc="left")  # fmt: skip
    style(top)
    top.grid(False, axis="y")

    x = torch.randn(4096, 1024, generator=torch.Generator().manual_seed(0))
    bins = torch.linspace(-12, 0, 97)
    for name in ("mxfp8", "mxfp6_e3m2", "mxfp6_e2m3", "mxfp4", "nvfp4"):
        fmt = mx.FORMATS[name]
        q = mx.quantize(x, fmt)
        y = mx.dequantize(q) * q.tensor_decode
        rel = ((y - x.double()).abs() / x.double().abs()).clamp(min=2.0**-12).log2()
        hist = torch.histogram(rel.float().flatten(), bins=bins).hist / rel.numel()
        bottom.step(bins[:-1], hist, where="post", color=COLOR[name], linewidth=1.6,
                    label=f"{LABEL[name]} (median {2 ** rel.median():.1e})")  # fmt: skip
    bottom.set_xlabel("log2 relative error of one element, N(0,1) data", fontsize=9, color=MUTED)
    bottom.set_ylabel("fraction of elements", fontsize=9, color=MUTED)
    bottom.legend(frameon=False, fontsize=8)
    style(bottom)
    fig.tight_layout()
    fig.savefig(IMG / "training-formats.png", dpi=150)


def read_log(path: Path):
    train, evals = [], []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r["event"] == "train":
            train.append(r)
        elif r["event"] == "eval":
            evals.append(r)
    return train, {r["step"]: r for r in evals}  # resumed chunks may repeat an eval


def curves(runs: Path, out: str) -> None:
    """Validation loss against tokens per run, and the gap to BF16."""
    logs = {p.parent.name: read_log(p) for p in sorted(runs.glob("*/log.jsonl"))}
    fig, (ax, gap) = plt.subplots(2, 1, figsize=(8, 6.4), height_ratios=[1.4, 1], sharex=True)
    base = logs.get("bf16", (None, {}))[1]
    for name, (_, evals) in logs.items():
        fmt = name.removesuffix("_sr")
        steps = sorted(evals)
        tokens = [evals[s]["tokens"] / 1e6 for s in steps]
        loss = [evals[s]["val_loss"] for s in steps]
        ls = "--" if name.endswith("_sr") else "-"
        label = LABEL.get(fmt, fmt) + (" + SR" if name.endswith("_sr") else "")
        ax.plot(tokens, loss, ls, color=COLOR.get(fmt, INK), linewidth=2, label=label)
        shared = [s for s in steps if s in base and s > 0]
        if name != "bf16" and shared:
            pct = [100 * (evals[s]["val_loss"] / base[s]["val_loss"] - 1) for s in shared]
            gap.plot([evals[s]["tokens"] / 1e6 for s in shared], pct, ls,
                     color=COLOR.get(fmt, INK), linewidth=2)  # fmt: skip
    ax.set_ylabel("validation loss", fontsize=9, color=MUTED)
    ax.legend(frameon=False, fontsize=8)
    lo = min(min(e["val_loss"] for e in ev.values()) for _, ev in logs.values())
    ax.set_ylim(lo - 0.1, lo + 2.5)
    gap.axhline(0, color=MUTED, linewidth=1)
    gap.set_ylabel("% above BF16", fontsize=9, color=MUTED)
    gap.set_xlabel("training tokens (millions)", fontsize=9, color=MUTED)
    for a in (ax, gap):
        style(a)
    fig.tight_layout()
    fig.savefig(IMG / f"{out}.png", dpi=150)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("figure", choices=["formats", "curves"])
    p.add_argument("--runs", type=Path)
    p.add_argument("--out", default="training-loss-curves")
    args = p.parse_args()
    IMG.mkdir(parents=True, exist_ok=True)
    if args.figure == "formats":
        formats()
    else:
        curves(args.runs, args.out)


if __name__ == "__main__":
    main()
