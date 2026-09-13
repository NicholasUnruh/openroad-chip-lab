# FP32 training tile

`training_tile` is a synthesizable 4×4 matrix engine: one FP32 fused multiply-add unit, three 16-word register banks, and a hardware sequencer. It computes `C = op(A) × op(B) + C` in 64 compute clocks. This is a training compute prototype; the host supplies data and schedules operations.

```sh
make sim-training
make sim-training-netlist
make layout-training
klayout build/layout-training/training_tile.gds
```

## Training operations

| Operation | A | B | Initial C | Transpose |
| --- | --- | --- | --- | --- |
| Forward | X | W | Zero | Neither |
| Weight gradient | X | dY | Zero or previous gradient | A |
| Input gradient | dY | W | Zero | B |
| SGD update | −learning_rate × identity | dW | W | Neither |

## Interface

All inputs are synchronous to `clk`. Matrices use row-major IEEE binary32 words; address is `4*row+column`.

| Signals | Contract |
| --- | --- |
| `rst` | Active-high synchronous reset; aborts work and clears control/flags. Operand/result banks retain unspecified or partial contents. |
| `write_enable[2:0]` | Idle write mask: bit 0=A, bit 1=B, bit 2=C. Multiple banks can receive the same word. |
| `write_addr[3:0]`, `write_data[31:0]` | Bank address and raw FP32 bits. Initialize all A/B words before use, and C when accumulating. |
| `start` | Pulse for one clock while idle. Takes priority over writes. Busy cycles ignore starts/writes. |
| `transpose_a`, `transpose_b` | Latched on start; selects each operand's transpose. |
| `accumulate` | On start, retain C when set; otherwise clear C to positive zero. |
| `busy`, `done` | Start asserts busy. The 64th following rising edge clears busy and pulses done. |
| `read_addr[3:0]`, `read_data[31:0]` | Asynchronous C read; completed results remain stable until a write/new job. Busy reads expose intermediate results. |
| `exception_flags[4:0]` | Sticky per-job invalid, divide-by-zero, overflow, underflow, inexact; clear on reset/start. FMA cannot raise divide-by-zero. Exceptions do not abort a job. |

## Arithmetic and verification

[HardFloat](../third_party/hardfloat/README.md) implements FP32 multiplication/addition with one rounding per FMA, nearest-even, gradual underflow, tininess after rounding, and canonical NaNs. Each four-term dot product rounds after each FMA; the whole dot product is not exact accumulation.

RTL and synthesized-netlist tests compare with host `std::fma(float, float, float)`. They cover numerical boundaries, all modes, command rejection, abort/restart, and finite-difference checks of both gradients. A dense-layer squared-error workload uses the tile for forward/backward products and SGD; host software computes loss and dY.

See [verified numerical and physical results](training-results.md) and [fabrication gates](physical-flow.md). This prototype does not establish general model convergence or fabrication readiness.

## Next milestones

Pipeline the FMA, add parallel lanes and SRAM-backed tile reuse, and evaluate BF16 operands with FP32 accumulation against this baseline. Full-model execution also needs larger-matrix scheduling, nonlinear operations, memory movement and software integration.
