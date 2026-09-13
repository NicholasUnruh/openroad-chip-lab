import json
import sys


def check(path):
    with open(path) as source:
        module = json.load(source)["modules"]["training_tile_v3"]
    multipliers = [
        cell for cell in module["cells"].values() if cell["type"] == "$mul"
    ]
    if len(multipliers) != 4:
        raise ValueError(f"expected four multipliers, found {len(multipliers)}")
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
    expected = {"engine.a_mem": (16, 32), "engine.b_mem": (16, 32)}
    expected.update({f"engine.lanes[{lane}].c_bank": (32, 8) for lane in range(4)})
    actual = {name: (memory["width"], memory["size"]) for name, memory in memories.items()}
    if actual != expected:
        raise ValueError(f"unexpected operand/accumulator storage: {actual}")
    print("PASS: four unsigned 8x8 -> 16 multipliers; BF16 A/B, FP32 C; two 128-byte banks")


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("usage: python3 tb/check_bf16_structure.py <yosys-json>")
        check(sys.argv[1])
    except (ValueError, KeyError, OSError) as error:
        print(f"FAIL: {error}")
        sys.exit(1)
