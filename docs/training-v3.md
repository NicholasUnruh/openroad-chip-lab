# Native BF16 training accelerator

`training_tile_v3` is the compact BF16 baseline: four BF16×BF16 lanes with fused FP32 accumulation, a 4×4 sequencer, and two independently selectable matrix buffers. Each job takes 19 compute clocks; host transfers can overlap computation. The default is now [V4](training-v4.md).

```sh
make sim-training-v3
make check-training-v3-structure
make sim-training-v3-netlist
make layout-training-v3
klayout build/layout-training-v3/training_tile_v3.gds
```

## Native arithmetic and storage

A/B registers store 16-bit BF16 values. Each lane multiplies two 8-bit normalized significands, retaining the exact 16-bit product. BF16 exponent/sign handling feeds the shared HardFloat alignment, normalization and rounding pipeline; the multiplier is physically narrowed. The structural check rejects operand widths other than 16 bits or multipliers other than four unsigned 8×8 units.

Each FMA rounds once to nearest-even FP32. C and outputs retain FP32 precision, including gradual underflow. BF16 subnormals are preserved; NaNs canonicalize to FP32 `0x7fc00000`. Products are not rounded to BF16 before accumulation. Host software quantizes new operands to BF16; no FP32 operand mode exists in V3.

Each buffer contains 32-byte A, 32-byte B and 64-byte C banks: 256 bytes total across both buffers, implemented in registers. The arithmetic pipeline and scheduler are shared with the [FP32 baseline](training-v2.md).

## Host interface

Matrices are row-major; element address is `4*row+column`. The [original control contract](training-tile.md#interface) applies with these changes:

| Signal/action | V3 contract |
| --- | --- |
| `job_bank` | Select buffer 0/1; latched with transpose/accumulate on accepted start. |
| `host_bank` | Select buffer for all host reads/writes independently of computation. |
| Write A/B | Mask bits 0/1. Even address required; data[15:0] writes that BF16 element, data[31:16] writes the following element. Mask 3 broadcasts to both operands. |
| Write C | Mask 4 writes one FP32 word at any address. |
| Invalid write | Odd operand address or combined operand/C mask rejects the entire transfer. |
| `write_ready` | Sample before the rising edge. A nonzero write mask transfers only when ready. Reset, accepted start and active-buffer ownership suppress readiness. |
| Start while busy | Ignored; valid writes to the inactive buffer still proceed. |
| Completion | The 19th rising edge after start clears busy and pulses done. The active buffer remains write-protected on that edge. |
| Reads | Asynchronous FP32 C from host_bank. Separate read/write ports permit concurrent transfers; sample reads before a simultaneous write edge. |

Reset aborts computation and clears control/flags, retaining unspecified or partial buffer contents. Exception flags combine across lanes and remain sticky until reset/start.

Consume a buffer's result before starting another job there. Host software owns initialization, scheduling and result lifetime; there is no command queue, SRAM or external bus adapter. See [measured results](training-v3-results.md).
