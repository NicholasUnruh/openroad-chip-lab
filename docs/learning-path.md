# Learning path

Start with the same block at each abstraction level. Read `rtl/mac8.v`, then predict the result of each operation in `tb/mac8_tb.v` before running `make sim`.

| Stage | Learn | Evidence |
| --- | --- | --- |
| Silicon fundamentals | MOSFETs, CMOS gates, standard cells, process layers, PDK views | Trace an inverter from transistors to its SKY130HD cell views |
| Digital logic | Bits, signedness, gates, adders, multipliers | Explain the 8 x 8 product and 24-bit accumulator |
| Sequential logic | Flip-flops, clock edges, reset, enable | `make sim` passes reset, hold, accumulation and wrap cases |
| RTL quality | Synthesis subset, inferred hardware, latches | `make synth` reports mapped cells without latches |
| Timing | Clock period, setup, hold, input/output budgets | Read `constraints/sky130hd.sdc`; inspect a timing path |
| Physical design | Floorplan, placement, clock tree, routing | Inspect OpenROAD reports and final database |
| Signoff | Functional tests, formal equivalence, DRC/LVS, parasitics, power | Record independent checks before a fabrication claim |
| Chip integration | IO pads, ESD, power, package pins and test access | Build a complete die and a fabrication acceptance checklist |

For the theory alongside this lab, use [MIT 6.004 Computation Structures](https://ocw.mit.edu/courses/6-004-computation-structures-spring-2017/) from MOS devices through digital systems. [Nand2Tetris Part I](https://www.nand2tetris.org/course) provides guided gate, ALU and CPU exercises. Then follow the [OpenROAD flow tutorial](https://openroad-flow-scripts.readthedocs.io/en/latest/tutorials/FlowTutorial.html) for the physical-design stages.

First experiment: change `a * b` to `a + b`, update the expected testbench values, run simulation and synthesis, then compare cell count and timing. Restore the multiplier before continuing the main flow.

Next project: a small UART transmitter or timer core with a written interface specification and self-checking testbench.
