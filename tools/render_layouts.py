"""Render the routed SKY130HD GDS files in build/ to PNG with KLayout, at one fixed scale.

usage: klayout -zz -r tools/render_layouts.py

Reads build/layout*/<top>.gds. Writes build/renders/<name>.png (whole die at UM_PER_PX),
build/renders/<name>_zoom.png for V4 and V6 (a ZOOM_W x ZOOM_H µm window at ZOOM_UM_PER_PX,
placed where the logic is densest) and build/renders/layouts.json (top cell, die size, SRAM
macro box, scales). tools/plot_figures.py turns the renders into the figures in docs/img/.
Rerun after any `make layout*` to refresh.

The palette is our own: the PDK's sky130hd.lyp renders as a solid block at die scale and as
stipple noise on a dark background.
"""

import json
from pathlib import Path

import pya

ROOT = Path(__file__).resolve().parents[1]
# The main checkout owns build/; a worktree under .claude/worktrees/ reads it from there.
BUILD = next(
    p / "build" for p in [ROOT, *ROOT.parents] if (p / "build/layout").is_dir()
)
OUT = BUILD / "renders"
UM_PER_PX = 1.25
ZOOM_W, ZOOM_H, ZOOM_UM_PER_PX = 30.0, 20.0, 0.02
BACKGROUND = "#0a0b12"
DESIGNS = [
    ("v1", "layout-training/training_tile.gds"),
    ("v2", "layout-training-v2/training_tile_v2.gds"),
    ("v3", "layout-training-v3/training_tile_v3.gds"),
    ("v4", "layout-training-v4/training_tile_v4.gds"),
    ("v6", "layout-training-v6/training_tile_v6.gds"),
    ("mac8", "layout/mac8.gds"),
]
# Name, SKY130 GDS layer/datatype, colour, then the dither pattern for the zoom and die views
# (0 solid, 2+ stipples, None hidden). Drawn bottom to top. The die view hides the device
# layers and met1 rails, which at 1.25 µm/px only repeat the cell rows.
PALETTE = [
    ("diff", 65, 20, "#2f6f4f", 0, None),
    ("poly", 66, 20, "#b04848", 0, None),
    ("li1", 67, 20, "#6a5acd", 2, None),
    ("met1", 68, 20, "#3d8bd9", 3, None),
    ("met2", 69, 20, "#d98c3d", 3, 0),
    ("met3", 70, 20, "#4fbfa8", 4, 0),
    ("met4", 71, 20, "#c060c0", 5, 0),
    ("met5", 72, 20, "#e8d06a", 9, 0),
]
SRAM_LAYER = (1000, 0)  # synthetic outline of the SRAM macro instance
SRAM_COLOUR = "#f07a45"


def set_layers(view, zoom, box):
    view.clear_layers()
    for name, layer, datatype, colour, zoom_dither, die_dither in PALETTE:
        dither = zoom_dither if zoom else die_dither
        if dither is None:
            continue
        node = pya.LayerPropertiesNode()
        node.source = f"{layer}/{datatype}@1"
        node.name = name
        node.fill_color = node.frame_color = int(colour[1:], 16)
        node.dither_pattern = dither
        view.insert_layer(view.end_layers(), node)
    if box:
        node = pya.LayerPropertiesNode()
        node.source = f"{SRAM_LAYER[0]}/{SRAM_LAYER[1]}@1"
        node.name = "sram"
        node.fill_color = node.frame_color = int(SRAM_COLOUR[1:], 16)
        node.dither_pattern = 1
        node.width = 3
        view.insert_layer(view.end_layers(), node)


def sram_box(top):
    for inst in top.each_inst():
        if inst.cell.name.startswith("sky130_sram"):
            return inst.cell.name, inst.dbbox()
    return None, None


def densest_window(layout, top, die, box):
    """ZOOM_W x ZOOM_H window, on a 12 x 12 grid, with the most poly shapes outside the SRAM."""
    poly = layout.find_layer(66, 20)
    best = (-1, None)
    for i in range(12):
        for j in range(12):
            x = die.left + ZOOM_W + (die.width() - 3 * ZOOM_W) * i / 11
            y = die.bottom + ZOOM_H + (die.height() - 3 * ZOOM_H) * j / 11
            window = pya.DBox(x, y, x + ZOOM_W, y + ZOOM_H)
            if box and window.overlaps(box):
                continue
            count = 0
            shapes = top.begin_shapes_rec_touching(poly, window)
            while not shapes.at_end() and count < 100000:
                count += 1
                shapes.next()
            best = max(best, (count, (x, y)), key=lambda b: b[0])
    x, y = best[1]
    return pya.DBox(x, y, x + ZOOM_W, y + ZOOM_H)


def render(name, gds, records):
    view = pya.LayoutView()
    view.set_config("background-color", BACKGROUND)
    view.set_config("grid-visible", "false")
    view.set_config("text-visible", "false")
    view.set_config("cell-box-visible", "false")
    index = view.load_layout(str(gds), True)
    layout = view.cellview(index).layout()
    top = layout.top_cell()
    die = top.dbbox()
    macro, box = sram_box(top)
    if box:
        top.shapes(layout.layer(*SRAM_LAYER)).insert(box)
    view.max_hier()
    set_layers(view, False, box)
    view.zoom_box(die)
    view.save_image(
        str(OUT / f"{name}.png"),
        round(die.width() / UM_PER_PX),
        round(die.height() / UM_PER_PX),
    )
    record = {
        "gds": str(gds.relative_to(BUILD.parent)),
        "top": top.name,
        "die_um": [round(die.width(), 3), round(die.height(), 3)],
        "um_per_px": UM_PER_PX,
        "png": f"{name}.png",
        "sram": {"cell": macro, "box_um": [box.left, box.bottom, box.right, box.top]}
        if box
        else None,
    }
    if name in ("v4", "v6"):
        window = densest_window(layout, top, die, box)
        set_layers(view, True, None)
        view.zoom_box(window)
        size = (round(ZOOM_W / ZOOM_UM_PER_PX), round(ZOOM_H / ZOOM_UM_PER_PX))
        view.save_image(str(OUT / f"{name}_zoom.png"), *size)
        record["zoom"] = {
            "png": f"{name}_zoom.png",
            "box_um": [
                round(v, 3)
                for v in (window.left, window.bottom, window.right, window.top)
            ],
            "um_per_px": ZOOM_UM_PER_PX,
        }
    records[name] = record
    view._destroy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = {
        "layers": {p[0]: f"{p[1]}/{p[2]} {p[3]}" for p in PALETTE},
        "sram_outline": SRAM_COLOUR,
    }
    for name, relative in DESIGNS:
        gds = BUILD / relative
        if gds.exists():
            render(name, gds, records)
            print(name, records[name]["die_um"], records[name].get("zoom", ""))
    (OUT / "layouts.json").write_text(json.dumps(records, indent=1) + "\n")


main()
