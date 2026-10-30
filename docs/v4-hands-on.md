# V4 hands-on lab

The V4 core is synthesizable Verilog (`.v`); the small teaching testbench is SystemVerilog (`.sv`). Start at the top and follow one matrix product through the hardware.

| Read in order | What to find |
| --- | --- |
| [`rtl/training_tile_v4.v`](../rtl/training_tile_v4.v) | Pins and the three parameters that select two buffers, BF16 operands and 8×8 dimensions. |
| [`rtl/training_tile_engine.v`](../rtl/training_tile_engine.v) | A/B storage, banked B reads, eight generated C/FMA lanes, and the start/issue/drain/done controller. |
| [`rtl/fp32_accum_fma_pipe.v`](../rtl/fp32_accum_fma_pipe.v) | BF16-to-FP32 conversion, an 8×8-bit significand product, three pipeline registers and FP32 rounding. |
| [`tb/training_tile_v4_walkthrough.sv`](../tb/training_tile_v4_walkthrough.sv) | One matrix job: write A and B, pulse start, wait 67 clocks, read and check C. |

From the repository root, run:

```sh
make sim-training-v4-walkthrough
gtkwave build/training-v4-walkthrough.vcd
klayout build/layout-training-v4/training_tile_v4.gds
```

In GTKWave, add `clk`, `write_enable`, `write_addr`, `start`, `busy`, `done`, `read_addr`, and `read_data` from `dut`. Then add `issue_index`, `row`, `inner`, `active_bank`, and `lane_valid` from `dut.engine`, plus `result` from `dut.engine.lanes[0]`. Set data buses to hexadecimal. The first 64 writes load packed BF16 pairs of 1.0 into A and B. `start` clears C, then the engine issues 64 groups of eight FMAs. Three clocks drain the arithmetic pipeline; `done` pulses and every C value is FP32 8.0 (`0x41000000`).

Compare the waveform to the RTL: `always @(posedge clk)` describes registers changing on clock edges; `<=` is a nonblocking register update; `assign` and `always @*` describe combinational logic. The `generate for` loop creates eight hardware lanes at elaboration time, not a software loop that runs one lane at a time. The HardFloat modules supply floating-point decode, alignment and rounding; follow their ports before reading their internals.

First experiment: change only the B write data in the walkthrough testbench to `32'h4000_4000` (two BF16 values of 2.0). Predict the output, update the expected FP32 bits to `32'h4180_0000` (16.0), rerun the walkthrough, and inspect the changed lane result. Then run `make sim-training-v4` for the full RTL regression. To inspect the physical implementation, use KLayout's hierarchy and layer panels and zoom to the top cell; the GDS is a routed core, not a fabricated chip.
