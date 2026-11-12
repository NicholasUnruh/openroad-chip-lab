# SKY130 time unit is ns. Inputs and outputs are registered inside each harness.
create_clock -name clk -period 10 [get_ports clk]
set_input_delay 1 -clock clk [all_inputs -no_clocks]
set_output_delay 1 -clock clk [all_outputs]
