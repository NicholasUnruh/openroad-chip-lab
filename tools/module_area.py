"""SKY130HD standard-cell area of each submodule of the V4 and V6 tiles (hierarchical ABC mapping).

usage: python3 tools/module_area.py <sky130_fd_sc_hd__tt_025C_1v80.lib>

Synthesizes each tile without flattening, maps it to the liberty and writes
build/renders/module_area.json: area including submodules for every instance kind under the top,
plus the top's own logic. This is pre-placement (no buffering, resizing or fill), so totals sit
below the routed standard-cell area; the SRAM macro is a black box and listed with its LEF area.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = next(
    p / "build" for p in [ROOT, *ROOT.parents] if (p / "build/layout").is_dir()
)
HARDFLOAT = sorted(
    str(p.relative_to(ROOT)) for p in (ROOT / "third_party/hardfloat").glob("*.v")
)
HARDFLOAT += ["third_party/hardfloat/RISCV/HardFloat_specialize.v"]
TILES = {
    "training_tile_v4": [
        "rtl/training_tile_v4.v",
        "rtl/training_tile_engine.v",
        "rtl/fp32_accum_fma_pipe.v",
    ],
    "training_tile_v6": [
        "rtl/training_tile_v6.v",
        "rtl/fp_dot8_fp32.v",
        "rtl/fp32_round.v",
    ],
}
SRAM = "sky130_sram_1rw1r_128x256_8"
SRAM_AREA_UM2 = 722109  # macro instance area in build/layout-training-v6/metrics.json


def stat(top, lib):
    script = (
        f"read_verilog -lib tb/{SRAM}.v; read_verilog -Ithird_party/hardfloat -Ithird_party/hardfloat/RISCV "
        f"{' '.join(TILES[top] + HARDFLOAT)}; synth -top {top}; dfflibmap -liberty {lib}; "
        f"abc -liberty {lib}; opt_clean; tee -o /dev/stdout stat -liberty {lib}"
    )
    out = subprocess.run(
        ["yosys", "-q", "-p", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=3600,
        check=True,
    ).stdout
    modules, name = {}, None
    for line in out.split("=== design hierarchy ===")[0].splitlines():
        if m := re.match(r"=== (\S+) ===", line):
            name = m.group(1)
            modules[name] = {"local": 0.0, "children": {}}
        elif (
            name
            and (m := re.match(r"\s+(\S+)\s+(\d+)$", line))
            and not m.group(1).startswith("sky130_fd_sc")
        ):
            modules[name]["children"][m.group(1)] = int(m.group(2))
        elif name and (
            m := re.match(r"\s+Chip area for module '(\S+)': ([\d.]+)", line)
        ):
            modules[name]["local"] = float(m.group(2))
    return modules


def total(modules, name):
    if name == SRAM:
        return SRAM_AREA_UM2
    if name not in modules:
        return 0.0
    mod = modules[name]
    return mod["local"] + sum(n * total(modules, c) for c, n in mod["children"].items())


def plain(name):
    return name.split("\\")[1] if name.startswith("$paramod") else name


def breakdown(modules, name):
    """Area of each child kind; a single wrapper instance (V4's engine) is opened up."""
    parts = [
        {
            "module": f"{plain(name)} (own logic)",
            "count": 1,
            "area_um2": round(modules[name]["local"], 1),
        }
    ]
    for child, count in modules[name]["children"].items():
        if count == 1 and modules.get(child, {}).get("children"):
            parts += breakdown(modules, child)
        else:
            parts.append(
                {
                    "module": plain(child),
                    "count": count,
                    "area_um2": round(count * total(modules, child), 1),
                }
            )
    return parts


def main():
    lib = sys.argv[1]
    result = {}
    for top in TILES:
        modules = stat(top, lib)
        result[top] = breakdown(modules, top)
        print(top, result[top])
    (BUILD / "renders").mkdir(exist_ok=True)
    (BUILD / "renders/module_area.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
