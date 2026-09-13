# Four-lane FP32 training tile

`training_tile_v2` computes a 4×4 matrix product using four pipelined FP32 fused multiply-add lanes. It preserves the scalar tile's per-output arithmetic order and finishes in 19 compute clocks after the accepted start.

```sh
make sim-training-v2
make sim-training-v2-netlist
make layout-training-v2
klayout build/layout-training-v2/training_tile_v2.gds
```

## Datapath and scheduling

Each lane owns one output column. A broadcasts to all four lanes; each lane selects its B element and C accumulator. The sequencer interleaves four output rows for each inner-product index. An accumulator's previous update completes before its next read.

| Stage | Work |
| --- | --- |
| 1 | Recode operands, align C, register HardFloat pre-multiply state |
| 2 | Multiply significands and add aligned C without rounding |
| 3 | Normalize and register raw result and exception state |
| 4 | Round to FP32 and write the lane's C bank |

Sixteen issue clocks perform 64 FMAs; three drain clocks complete the last results. All lanes accept work together. Exceptions combine across lanes and remain sticky until reset/start.

The A, B and C banks hold 192 bytes total in registers. C is partitioned by output column. Existing inputs can be reused across jobs; the host port still transfers one 32-bit word per clock and cannot load operands while busy. There are no SRAM macros or overlapping load/compute buffers.

## Interface and arithmetic

Use the [scalar interface contract](training-tile.md#interface), with one latency change: the **19th** rising edge after start clears `busy` and pulses `done`. Transpose, accumulation, reset/abort, bank writes and result reads retain their meanings.

The pipeline carries full intermediate precision and rounds once per FMA: nearest-even FP32 with gradual underflow, tininess after rounding and canonical NaNs. Four-term dot products still round after each FMA. Pipelining preserves the scalar numerical result.

The shared testbench verifies forward products, dW, dX and SGD with the [same reference arithmetic](training-tile.md#arithmetic-and-verification). See [physical results and transfer benchmarks](training-v2-results.md) for measured cycle counts, timing estimates and limits.
