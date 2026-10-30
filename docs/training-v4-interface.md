# V4 host interface

`training_tile_v4` has one 32-bit write port and one 32-bit read port over two buffers, so the host transfers one buffer while the other computes. All inputs are synchronous to `clk`. Matrices are row-major; element address is `8*row+column`, range 0–63.

## Control and data signals

| Signal/action | Contract |
| --- | --- |
| `rst` | Active-high synchronous. Aborts computation and clears control/flags; buffers retain unspecified or partial contents. |
| `start` | Pulse one clock while idle. Wins over a simultaneous write. Ignored while busy; valid writes to the inactive buffer still proceed. |
| `job_bank` | Selects buffer 0/1 for computation; latched on accepted start. |
| `host_bank` | Selects the buffer for all host reads/writes, independently of computation. |
| `transpose_a`, `transpose_b` | Latched on accepted start. |
| `accumulate` | On accepted start, retains C when set; otherwise clears the job buffer's C to +0. |
| `write_enable[2:0]` | Bit 0=A, bit 1=B, bit 2=C. |
| Write A/B | Even address required. `write_data[15:0]` writes that BF16 element, `[31:16]` the following one. Mask 3 broadcasts to both operands. |
| Write C | Mask 4 writes one FP32 word at any address. |
| Invalid write | An odd operand address or a combined operand/C mask rejects the entire transfer. |
| `write_ready` | Sample before the rising edge. A nonzero mask transfers only when ready. Reset, accepted start and active-buffer ownership suppress readiness. |
| `busy`, `done` | Accepted start asserts `busy`. The 67th following rising edge clears it and pulses `done`. The active buffer stays write-protected on that edge. |
| `read_addr`, `read_data` | Asynchronous FP32 C from `host_bank`. Sample before a simultaneous write edge. Results stay stable until a write or new job; reading the active buffer exposes intermediate values. |
| `exception_flags[4:0]` | Invalid, divide-by-zero, overflow, underflow, inexact. Combined across lanes, sticky until reset/start. FMAs never raise divide-by-zero. Exceptions do not abort a job. |

## Transfer budget and buffer use

| Item | V4 |
| --- | --- |
| A/B preload | 64 clocks per buffer |
| C initialization/readout | 64 clocks each |
| Compute / earliest start interval | 67 / 68 clocks |

Alternate `job_bank` between jobs. During compute, set `host_bank` to the other buffer, read its previous C and preload its next A/B concurrently. Consume C before reusing that buffer. A/B preloading fits within computation; externally supplied C needs another 64 writes. Host software owns initialization, scheduling and result lifetime.

## Training operations

| Operation | A | B | Initial C | Transpose |
| --- | --- | --- | --- | --- |
| Forward | X | W | Zero | Neither |
| Weight gradient | X | dY | Zero or previous gradient | A |
| Input gradient | dY | W | Zero | B |
| SGD update | −learning_rate × identity | dW | W | Neither |
