# Eight-lane BF16 accelerator

`training_tile_v4` computes an 8×8 matrix product with eight native BF16 lanes, fused FP32 accumulation and two buffers. It doubles operand reuse per output column compared with the retired four-lane BF16 tile.

`rtl/training_tile_engine.v` keeps its earlier four-lane, unbuffered FP32 parameter defaults; `training_tile_v4` overrides all three. Leave them in place: these sources are vendored unmodified into the FPGA build.

```sh
make sim-training-v4
make check-training-v4-structure
make sim-training-v4-netlist
make layout-training-v4
klayout build/layout-training-v4/training_tile_v4.gds
```

## Arithmetic and scheduling

A/B storage and multiplication use BF16 exclusively. Each lane has an unsigned 8×8-bit significand multiplier with an exact 16-bit product. BF16 exponent/sign handling feeds a shared HardFloat alignment, normalization and rounding pipeline. Each FMA rounds once to nearest-even FP32; C and outputs retain FP32 precision, including gradual underflow with tininess after rounding. An eight-term sum rounds after each FMA, so it is not exact accumulation. BF16 subnormals are preserved; NaNs canonicalize to FP32 `0x7fc00000`. Products are not rounded to BF16 before accumulation. Host software quantizes new operands to BF16; no FP32 operand mode exists. The structural check rejects operand widths other than 16 bits or multipliers other than eight unsigned 8×8 units.

Eight rows interleave accumulator dependencies. The sequencer issues 64 groups of eight FMAs, followed by three drain clocks: 512 FMAs, or 1,024 FLOPs, per job. The 67th rising edge after accepted start clears `busy` and pulses `done`; another start can be accepted on the following edge. Exception flags combine across all eight lanes.

Each buffer contains 128-byte A, 128-byte B and 256-byte C banks. Total matrix storage is 1,024 bytes in registers. Pipeline registers are additional. Larger matrices retain partial C across K chunks and pad incomplete edge tiles with zeros.

B uses eight physical banks selected by `row XOR column`. Both normal and transposed reads access one element per bank, reducing read-selection logic. Packed BF16 pairs write two banks with one write each. Fixed even/odd A write addresses remove redundant write multiplexers.

## Host interface

The [host interface](training-v4-interface.md) owns signals, the write/read contract, the transfer budget and training operations.

There is no DMA, command queue, SRAM macro or external bus adapter. See [verification and measured tradeoffs](training-v4-results.md) and the [fabrication gates](physical-flow.md).
