"""Build the figures in docs/img/ from layout renders, OpenROAD reports and numerics outputs.

usage: uv run --with matplotlib --with pillow python tools/plot_figures.py [--site <dir>]

Inputs (all under the main checkout's build/):
  renders/          klayout -zz -r tools/render_layouts.py
  renders/training_tile_v6/   tools/physical_views.sh
  renders/module_area.json    python3 tools/module_area.py <liberty>
  layout*/metrics.json        make layout, make layout-training*
  numerics/train_*.txt        make numerics
and the ASAP7 tables in docs/numerics-results.md. --site also writes WebP copies and the
figure data as JSON to <dir>.
"""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUILD = next(
    p / "build" for p in [ROOT, *ROOT.parents] if (p / "build/layout").is_dir()
)
RENDERS = BUILD / "renders"
IMG = ROOT / "docs/img"
sys.path.insert(0, str(ROOT / "tools/numerics"))
import report  # noqa: E402

report.ROOT = BUILD.parent

BG, INK, MUTED, GRID, ACCENT = "#0a0b12", "#f2f2f0", "#9a9ca6", "#2a2c36", "#f07a45"
BLUE, AQUA, VIOLET = "#3987e5", "#199e70", "#9085e9"
plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)
# Layout directory per tile, and the one-line description of what each version changed.
VERSIONS = {
    "v1": ("layout-training", "V1 · scalar FP32 FMA"),
    "v2": ("layout-training-v2", "V2 · four FP32 lanes"),
    "v3": ("layout-training-v3", "V3 · four BF16 lanes"),
    "v4": ("layout-training-v4", "V4 · eight BF16 lanes"),
    "v6": ("layout-training-v6", "V6 · 4×8-term fused dots + SRAM"),
    "mac8": ("layout", "MAC8 · 8-bit warm-up"),
}


def metrics(name):
    m = json.loads((BUILD / VERSIONS[name][0] / "metrics.json").read_text())
    return {
        "stdcell_um2": m["finish__design__instance__area__stdcell"],
        "die_um2": m["finish__design__die__area"],
        "setup_ws_ns": m["finish__timing__setup__ws"],
        "hold_ws_ns": m["finish__timing__hold__ws"],
        "fmax_mhz": round(m["finish__timing__fmax"] / 1e6, 2),
        "stdcells": m.get("finish__design__instance__count__stdcell"),
    }


def save_webp(image, path):
    """Site copy: WebP quality 82, halved in steps of 0.85 until it is under 250 KB."""
    while True:
        image.save(path, "WEBP", quality=82, method=6)
        if path.stat().st_size < 250_000:
            return
        image = image.resize(
            (int(image.width * 0.85), int(image.height * 0.85)), Image.LANCZOS
        )


def save(fig, name, site, data=None, data_name=None):
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / f"{name}.png", dpi=100)
    plt.close(fig)
    # 128-colour palette PNGs keep the committed images to a few hundred KB.
    image = Image.open(IMG / f"{name}.png").convert("RGB")
    image.quantize(colors=128, method=Image.Quantize.MEDIANCUT).save(
        IMG / f"{name}.png", optimize=True
    )
    if site:
        save_webp(image, site / f"{name}.webp")
        if data is not None:
            (site / f"{data_name or name}.json").write_text(
                json.dumps(data, separators=(",", ":")) + "\n"
            )


def scale_bar(ax, x, y, length, label):
    ax.plot([x, x + length], [y, y], color=INK, lw=2, solid_capstyle="butt")
    ax.text(x + length * 1.05, y, label, ha="left", va="center", color=INK, fontsize=10)


def die_figure(name, layouts, site):
    """One die at its render scale (UM_PER_PX in render_layouts.py), with a label and scale bar."""
    rec = layouts[name]
    w, h = rec["die_um"]
    img = Image.open(RENDERS / rec["png"])
    m = metrics(name)
    margin = 64
    fig = plt.figure(figsize=((img.width) / 100, (img.height + margin) / 100))
    ax = fig.add_axes([0, 0, 1, img.height / (img.height + margin)])
    ax.imshow(img, extent=(0, w, 0, h), interpolation="nearest")
    ax.set_axis_off()
    fig.text(
        0.01,
        1 - 26 / (img.height + margin),
        f"{VERSIONS[name][1]}",
        fontsize=13,
        weight="bold",
    )
    fig.text(
        0.01,
        1 - 50 / (img.height + margin),
        f"{w:,} × {h:,} µm · {m['stdcell_um2']:,.0f} µm² standard cells · {rec['um_per_px']} µm/px",
        fontsize=10,
        color=MUTED,
    )
    save(fig, f"layout-{name}", site)


def strip(layouts, site):
    names = ["v2", "v3", "v4", "v6"]
    gap = 120.0
    widths = [layouts[n]["die_um"][0] for n in names]
    height = max(layouts[n]["die_um"][1] for n in names)
    total = sum(widths) + gap * (len(names) - 1)
    fig = plt.figure(figsize=(16, 16 * (height + 330) / total))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, total)
    ax.set_ylim(-150, height + 180)
    ax.set_axis_off()
    x = 0.0
    data = []
    for n, w in zip(names, widths):
        rec, m = layouts[n], metrics(n)
        h = rec["die_um"][1]
        ax.imshow(
            Image.open(RENDERS / rec["png"]),
            extent=(x, x + w, 0, h),
            interpolation="antialiased",
        )
        ax.text(x, h + 95, VERSIONS[n][1], fontsize=12, weight="bold", va="bottom")
        ax.text(
            x,
            h + 30,
            f"{w:,.0f} × {h:,.0f} µm · {m['stdcell_um2'] / 1e3:,.0f}k µm² cells",
            fontsize=10,
            color=MUTED,
            va="bottom",
        )
        data.append(
            {"version": n, "label": VERSIONS[n][1], "die_um": rec["die_um"], **m}
        )
        if n != names[-1]:
            ax.annotate(
                "",
                xy=(x + w + gap * 0.85, h / 2),
                xytext=(x + w + gap * 0.15, h / 2),
                arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 2},
            )
        x += w + gap
    scale_bar(ax, 0, -110, 500, "500 µm — every die at the same scale")
    ax.text(
        total,
        -110,
        "SKY130HD routed cores · orange box: SRAM macro",
        ha="right",
        fontsize=10,
        color=MUTED,
        va="bottom",
    )
    save(fig, "layout-strip", site, data)


def zoom(name, layouts, site):
    rec = layouts[name]["zoom"]
    x0, y0, x1, y1 = rec["box_um"]
    img = Image.open(RENDERS / rec["png"])
    fig = plt.figure(figsize=(img.width / 100, (img.height + 70) / 100))
    ax = fig.add_axes([0, 0, 1, img.height / (img.height + 70)])
    ax.imshow(img, extent=(0, x1 - x0, 0, y1 - y0), interpolation="nearest")
    ax.set_xlim(0, x1 - x0)
    ax.set_ylim(0, y1 - y0)
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0.6, 0.5), 6.2, 1.6, color=BG, alpha=0.85))
    ax.plot([1, 6], [1, 1], color=INK, lw=3)
    ax.text(3.5, 1.25, "5 µm", ha="center", va="bottom", fontsize=12)
    title = f"{VERSIONS[name][1].split(' · ')[0]}: {x1 - x0:.0f} × {y1 - y0:.0f} µm of placed standard cells and routing"
    fig.text(0.01, 1 - 30 / (img.height + 70), title, fontsize=14, weight="bold")
    fig.text(
        0.01,
        1 - 56 / (img.height + 70),
        "Layers, bottom to top: " + ", ".join(layouts["layers"]),
        fontsize=10,
        color=MUTED,
    )
    save(fig, f"layout-zoom-{name}", site)


def physical_views(site):
    folder = RENDERS / "training_tile_v6"
    panels = [
        ("density.png", "Placement density"),
        ("congestion.png", "Global-route congestion"),
        ("clock.png", "Clock nets and clock buffers"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.1))
    for ax, (file, title) in zip(axes, panels):
        ax.imshow(Image.open(folder / file))
        ax.set_title(title, loc="left")
        ax.set_axis_off()
    fig.suptitle(
        "V6 in OpenROAD: 1,700 µm die, SRAM macro at lower left (legend: % of capacity)",
        x=0.01,
        ha="left",
        fontsize=11,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.01, wspace=0.03)
    save(fig, "v6-physical-views", site)


def endpoint_slacks(path):
    slacks = []
    for line in path.read_text().splitlines():
        if m := re.search(r"\s(-?[\d.]+) \((MET|VIOLATED)\)$", line):
            slacks.append(float(m.group(1)))
    return slacks


def histogram(values, width):
    """{bin start: count} with bins of the given width."""
    counts = {}
    for v in values:
        start = round(width * (v // width), 2)
        counts[start] = counts.get(start, 0) + 1
    return {str(k): counts[k] for k in sorted(counts)}


def slack_histogram(site):
    folder = RENDERS / "training_tile_v6"
    setup, hold = (
        endpoint_slacks(folder / "endpoints_setup.txt"),
        endpoint_slacks(folder / "endpoints_hold.txt"),
    )
    v4, v6 = metrics("v4"), metrics("v6")
    fig, (a, b) = plt.subplots(1, 2, figsize=(16, 5.2))
    a.hist(setup, bins=60, color=BLUE, edgecolor=BG, linewidth=0.6)
    a.axvline(v6["setup_ws_ns"], color=ACCENT, lw=2)
    a.text(
        v6["setup_ws_ns"],
        a.get_ylim()[1] * 0.92,
        f"  V6 worst {v6['setup_ws_ns']:.2f} ns",
        color=INK,
    )
    a.axvline(v4["setup_ws_ns"], color=MUTED, lw=1.5, ls="--")
    a.text(
        v4["setup_ws_ns"],
        a.get_ylim()[1] * 0.80,
        f"  V4 worst {v4['setup_ws_ns']:.2f} ns",
        color=MUTED,
    )
    a.set_title(f"V6 setup slack, {len(setup):,} endpoints, 25 ns clock", loc="left")
    a.set_xlabel("slack (ns)")
    a.set_ylabel("endpoints")
    b.hist(hold, bins=60, color=AQUA, edgecolor=BG, linewidth=0.6)
    b.axvline(v6["hold_ws_ns"], color=ACCENT, lw=2)
    b.text(
        v6["hold_ws_ns"],
        b.get_ylim()[1] * 0.92,
        f"  V6 worst {v6['hold_ws_ns']:.2f} ns",
        color=INK,
    )
    b.set_title(f"V6 hold slack, {len(hold):,} endpoints", loc="left")
    b.set_xlabel("slack (ns)")
    fig.tight_layout()
    data = {
        "bin_ns": 0.25,
        "v6_setup_counts": histogram(setup, 0.25),
        "v6_hold_counts": histogram(hold, 0.25),
        "v6_worst_setup_ns": v6["setup_ws_ns"],
        "v6_worst_hold_ns": v6["hold_ws_ns"],
        "v4_worst_setup_ns": v4["setup_ws_ns"],
        "endpoints": len(setup),
    }
    save(fig, "v6-slack-histogram", site, data)


MODULE_NAMES = {
    "training_tile_engine (own logic)": "engine: operand/result buffers, control",
    "fp32_accum_fma_pipe": "BF16 FMA lane",
    "training_tile_v6 (own logic)": "tile: operand flops, sequencer, control",
    "fp_dot8_fp32": "8-term dot unit",
    "sky130_sram_1rw1r_128x256_8": "4 KB SRAM macro",
}


def area_breakdown(site):
    areas = json.loads((RENDERS / "module_area.json").read_text())
    groups = {"training_tile_v4": "V4", "training_tile_v6": "V6"}
    colours = [BLUE, AQUA, VIOLET, "#c98500", "#d55181", "#e66767"]
    fig, ax = plt.subplots(figsize=(16, 4.4))
    legend = {}
    for row, (top, label) in enumerate(groups.items()):
        left = 0.0
        for i, part in enumerate(sorted(areas[top], key=lambda p: -p["area_um2"])):
            if part["area_um2"] <= 0:
                continue
            name = MODULE_NAMES.get(part["module"], part["module"]) + (
                f" ×{part['count']}" if part["count"] > 1 else ""
            )
            colour = (
                ACCENT
                if part["module"].startswith("sky130_sram")
                else colours[i % len(colours)]
            )
            ax.barh(
                row,
                part["area_um2"],
                left=left,
                color=colour,
                edgecolor=BG,
                linewidth=2,
                height=0.6,
            )
            if part["area_um2"] > 60000:
                ax.text(
                    left + 8000,
                    row,
                    f"{name}\n{part['area_um2'] / 1e3:,.0f}k µm²",
                    va="center",
                    fontsize=9.5,
                    color=BG if colour != BG else INK,
                )
            legend.setdefault(name, colour)
            left += part["area_um2"]
        ax.text(left + 8000, row, f"{left / 1e3:,.0f}k µm²", va="center", color=INK)
    ax.set_yticks(range(len(groups)), list(groups.values()))
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:,.0f}k")
    )
    ax.set_xlabel(
        "area (µm²): SKY130HD cells after Yosys/ABC mapping, before placement; SRAM macro at its LEF area"
    )
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(
        "Where the area goes: V4 lanes vs V6 dot-product units and scratchpad",
        loc="left",
    )
    fig.tight_layout()
    save(fig, "area-by-module", site, areas)


def numerics_table():
    """Area, fmax and energy rows of docs/numerics-results.md, keyed by point."""
    text = (ROOT / "docs/numerics-results.md").read_text()
    section = text.split("## Area, fmax and energy")[1].split("##")[0]
    rows = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 7 and cells[0].startswith("`"):
            point = cells[0].strip("`")
            macs, area, area_mac, fmax, mw, fj = (float(c) for c in cells[1:])
            rows[point] = {
                "macs_per_clk": macs,
                "area_um2": area,
                "area_per_mac_um2": area_mac,
                "fmax_mhz": fmax,
                "power_mw": mw,
                "fj_per_mac": fj,
                "gmacs_mm2": round(macs * fmax / 1e3 / (area * 1e-6)),
                "macs_nj": round(1e6 / fj),
            }
    return rows


# Direct labels (text, x and y offset in points); the BF16 frame-width variants share one label.
LABELS = {
    "dot32_mxfp8": ("MXFP8 32-term dot", 9, 0),
    "dot32_mxfp4": ("MXFP4 32-term dot (fails the gate)", -8, -16),
    "dot8_e4m3": ("FP8 E4M3 dot8", 9, 7),
    "dot8_e5m2": ("FP8 E5M2 dot8", 9, -7),
    "dot8_bf16": ("BF16 dot8, V6 unit (grey: frames 24–40)", 9, -9),
    "fma_bf16": ("BF16 FMA, V4 lane", 9, 0),
}


def numerics(site):
    ppa = numerics_table()
    train = report.training()
    fp32 = train["fp32"][0]
    points = []
    for harness, _, recipe in report.POINTS:
        point = harness[4:]
        row = {"point": point, "recipe": recipe, **ppa[point]}
        if recipe:
            row["test_loss"] = round(train[recipe][0], 4)
            row["loss_vs_fp32_pct"] = round((train[recipe][0] / fp32 - 1) * 100, 2)
            row["trains"] = train[recipe][0] <= fp32 * (1 + report.LOSS_TOLERANCE)
        points.append(row)
    gate = report.LOSS_TOLERANCE * 100
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    for ax, key, label in (
        (axes[0], "gmacs_mm2", "GMAC/s per mm² (routed fmax)"),
        (axes[1], "macs_nj", "MACs per nJ (simulated activity, 1 GHz)"),
    ):
        ax.axvspan(gate, 10, color=ACCENT, alpha=0.08, lw=0)
        ax.axvline(gate, color=ACCENT, lw=1.5, ls="--")
        ax.text(
            gate + 0.1,
            0.97,
            "1% training gate",
            transform=ax.get_xaxis_transform(),
            color=ACCENT,
            va="top",
        )
        for p in points:
            if "trains" not in p:
                continue
            frame = p["point"].startswith("dot8_bf16_f")
            colour = (MUTED if frame else BLUE) if p["trains"] else ACCENT
            ax.scatter(
                p["loss_vs_fp32_pct"],
                p[key],
                s=40 if frame else 80,
                color=colour,
                edgecolor=BG,
                linewidth=1.5,
                zorder=3,
            )
            if p["point"] in LABELS:
                text, dx, dy = LABELS[p["point"]]
                ax.annotate(
                    text,
                    (p["loss_vs_fp32_pct"], p[key]),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    fontsize=10,
                    va="center",
                    ha="right" if dx < 5 else "left",
                    color=INK,
                )
        ax.set_ylim(0, None)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
        )
        ax.set_xlim(-0.6, 10)
        ax.set_xlabel("final test loss vs FP32 (%), 64-unit MLP, 3 seeds")
        ax.set_title(label, loc="left")
    fig.suptitle(
        "ASAP7 (predictive 7 nm) MAC datapaths: density and energy against training accuracy. "
        "dot8_int8 (inference only) is not plotted.",
        x=0.01,
        ha="left",
        fontsize=11,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(
        fig,
        "numerics-pareto",
        site,
        {"fp32_test_loss": round(fp32, 4), "gate_pct": gate, "points": points},
        "numerics",
    )
    return train


def loss_curves(site):
    curves = {}
    for path in sorted((BUILD / "numerics").glob("train_*.txt")):
        for line in path.read_text().splitlines():
            f = line.split()
            if f[0] in ("curve", "final"):
                curves.setdefault(f[1], {}).setdefault(int(f[3]), []).append(
                    float(f[5])
                )
    shown = [
        ("fp32", "FP32", INK),
        ("bf16_dot8", "BF16 dot8", BLUE),
        ("fp8_tensor", "FP8 per-tensor", AQUA),
        ("mxfp8", "MXFP8", VIOLET),
        ("mxfp4", "MXFP4", ACCENT),
    ]
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    data = {}
    for recipe, label, colour in shown:
        steps = sorted(curves[recipe])
        mean = [sum(curves[recipe][s]) / len(curves[recipe][s]) for s in steps]
        data[recipe] = {"step": steps, "test_loss_mean": [round(v, 5) for v in mean]}
        for ax in (a, b):
            ax.plot(steps, mean, color=colour, lw=2, label=label)
    a.set_title("Test loss during training (mean of 3 seeds)", loc="left")
    a.set_xlabel("SGD step")
    a.set_ylabel("test loss")
    a.legend()
    b.set_ylim(0.78, 1.0)
    b.set_xlim(1400, 3050)
    b.set_title("Last half, zoomed: MXFP4 stays above the others", loc="left")
    b.set_xlabel("SGD step")
    fig.tight_layout()
    save(fig, "numerics-loss-curves", site, data)


def main(argv):
    site = Path(argv[argv.index("--site") + 1]) if "--site" in argv else None
    if site:
        site.mkdir(parents=True, exist_ok=True)
    layouts = json.loads((RENDERS / "layouts.json").read_text())
    for name in VERSIONS:
        if name in layouts:
            die_figure(name, layouts, site)
    if site:
        dies = {
            n: {"label": VERSIONS[n][1], "die_um": layouts[n]["die_um"], **metrics(n)}
            for n in VERSIONS
            if n in layouts
        }
        (site / "layouts.json").write_text(json.dumps(dies, indent=1) + "\n")
        # Home-page thumbnail: the V6 die render, 640 px wide, centre-cropped to 640 x 400.
        thumb = Image.open(RENDERS / layouts["v6"]["png"]).convert("RGB")
        thumb = thumb.resize(
            (640, round(thumb.height * 640 / thumb.width)), Image.LANCZOS
        )
        top = (thumb.height - 400) // 2
        save_webp(thumb.crop((0, top, 640, top + 400)), site / "layout-v6-thumb.webp")
    strip(layouts, site)
    for name in ("v4", "v6"):
        zoom(name, layouts, site)
    physical_views(site)
    slack_histogram(site)
    area_breakdown(site)
    numerics(site)
    loss_curves(site)


if __name__ == "__main__":
    main(sys.argv)
