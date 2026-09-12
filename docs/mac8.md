# MAC8 interface

`mac8` is an unsigned, 8 x 8 multiply-accumulate core. On each rising edge of `clk`, `rst=1` clears `acc` to zero. Otherwise `en=1` adds `a*b` to `acc` modulo 2²⁴. With `en=0`, `acc` holds. Reset has priority over enable and is synchronous: it takes effect only at a clock edge.

| Port | Width | Meaning |
| --- | ---: | --- |
| `clk` | 1 | Rising-edge clock |
| `rst` | 1 | Synchronous active-high reset |
| `en` | 1 | Accumulate enable |
| `a`, `b` | 8 each | Unsigned operands |
| `acc` | 24 | Registered accumulated result |

The testbench drives each input on a falling edge and checks `acc` after the following rising edge. This avoids a testbench race with the register update. Inputs are assumed to be synchronous to `clk`; external pins and clock-domain crossing require later chip-integration work.
