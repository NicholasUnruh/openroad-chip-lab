# ASAP7 time unit is ps. Inputs and outputs are registered inside each harness.
create_clock -name clk -period 1000 [get_ports clk]
set_input_delay 100 -clock clk [all_inputs -no_clocks]
set_output_delay 100 -clock clk [all_outputs]
