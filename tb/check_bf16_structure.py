import json
import sys


def check(path, top="training_tile_v3"):
    dimension = {"training_tile_v3": 4, "training_tile_v4": 8, "training_tile_v5": 8}[top]
    with open(path) as source:
        module = json.load(source)["modules"][top]
    multipliers = [
        cell for cell in module["cells"].values() if cell["type"] == "$mul"
    ]
    if len(multipliers) != dimension:
        raise ValueError(f"expected {dimension} multipliers, found {len(multipliers)}")
    for cell in multipliers:
        if any(int(cell["parameters"][key], 2) for key in ("A_SIGNED", "B_SIGNED")):
            raise ValueError("significand multipliers must be unsigned")
        widths = tuple(
            int(cell["parameters"][key], 2)
            for key in ("A_WIDTH", "B_WIDTH", "Y_WIDTH")
        )
        if widths != (8, 8, 16):
            raise ValueError(f"expected native BF16 8x8 -> 16 multiply, found {widths}")
    memories = module["memories"]
    expected = {"engine.a_mem": (16, 2*dimension**2)}
    if dimension == 8:
        expected.update({f"engine.banked_b.banks[{bank}].mem": (16, 2*dimension) for bank in range(dimension)})
    else:
        expected["engine.b_mem"] = (16, 2*dimension**2)
    expected.update({f"engine.lanes[{lane}].c_bank": (32, 2*dimension) for lane in range(dimension)})
    actual = {name: (memory["width"], memory["size"]) for name, memory in memories.items()}
    if actual != expected:
        raise ValueError(f"unexpected operand/accumulator storage: {actual}")
    if top == "training_tile_v5":
        macros = [cell for cell in module["cells"].values() if cell["type"] == "sky130_sram_1rw1r_128x256_8"]
        if len(macros) != 1:
            raise ValueError(f"expected one scratchpad SRAM macro, found {len(macros)}")
    print(f"PASS: {dimension} unsigned 8x8 -> 16 multipliers; BF16 A/B, FP32 C; two {8*dimension**2}-byte banks")


if __name__ == "__main__":
    try:
        if len(sys.argv) not in (2, 3):
            raise ValueError("usage: python3 tb/check_bf16_structure.py <yosys-json> [training_tile_v3|training_tile_v4|training_tile_v5]")
        check(*sys.argv[1:])
    except (ValueError, KeyError, OSError) as error:
        print(f"FAIL: {error}")
        sys.exit(1)
