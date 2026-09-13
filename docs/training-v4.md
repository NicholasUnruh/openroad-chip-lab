# Eight-lane BF16 accelerator

`training_tile_v4` computes an 8×8 matrix product with eight native BF16 lanes, fused FP32 accumulation and two buffers. It doubles operand reuse per output column compared with the [four-lane BF16 baseline](training-v3.md). The shared engine retains V2/V3 dimensions by default.

```sh
make sim-training-v4
make check-training-v4-structure
make sim-training-v4-netlist
make layout-training-v4
klayout build/layout-training-v4/training_tile_v4.gds
```

## Arithmetic and scheduling

A/B storage and multiplication use BF16 exclusively. Each lane has an unsigned 8×8-bit significand multiplier with an exact 16-bit product. The shared arithmetic pipeline retains FP32 accumulation/results and one nearest-even rounding per FMA. BF16 subnormals, canonical NaNs and exception flags follow the [V3 arithmetic contract](training-v3.md#native-arithmetic-and-storage). Host software quantizes new operands to BF16.

Eight rows interleave accumulator dependencies. The sequencer issues 64 groups of eight FMAs, followed by three drain clocks: 512 FMAs, or 1,024 FLOPs, per job. The 67th rising edge after accepted start clears `busy` and pulses `done`; another start can be accepted on the following edge. Exception flags combine across all eight lanes.

Each buffer contains 128-byte A, 128-byte B and 256-byte C banks. Total matrix storage is 1,024 bytes in registers. Pipeline registers are additional. Larger matrices retain partial C across K chunks and pad incomplete edge tiles with zeros.

B uses eight physical banks selected by `row XOR column`. Both normal and transposed reads access one element per bank, reducing read-selection logic. Packed BF16 pairs write two banks with one write each. Fixed even/odd A write addresses remove redundant write multiplexers.

## Host interface

The [V3 buffer and transfer contract](training-v3.md#host-interface) applies with these dimensions:

| Item | V4 |
| --- | --- |
| Element address | Six bits, `8*row+column`, range 0–63 |
| Data ports | Independent 32-bit write and 32-bit read |
| A/B write | Two packed BF16 values at an even element address |
| C write/read | One FP32 element |
| A/B preload | 64 clocks per buffer |
| C initialization/readout | 64 clocks each |
| Compute / earliest start interval | 67 / 68 clocks |

Alternate `job_bank` between jobs. During compute, set `host_bank` to the other buffer, read its previous C and preload its next A/B concurrently. Consume C before reusing that buffer. A/B preloading fits within computation; externally supplied C needs another 64 writes. Reset aborts computation and retains unspecified or partial buffer contents.

There is no DMA, command queue, SRAM macro or external bus adapter. See [verification and measured tradeoffs](training-v4-results.md) and the [fabrication gates](physical-flow.md).
