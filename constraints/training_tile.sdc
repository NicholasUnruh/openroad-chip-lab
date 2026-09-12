create_clock -name clk -period 100 [get_ports clk]
set_input_delay 5 -clock clk [all_inputs -no_clocks]
set_output_delay 5 -clock clk [all_outputs]
