.PHONY: sim synth layout

sim:
	mkdir -p build
	iverilog -g2012 -s mac8_tb -o build/mac8_tb.vvp rtl/mac8.v tb/mac8_tb.v
	vvp build/mac8_tb.vvp

synth:
	yosys -Q -T -p 'read_verilog rtl/mac8.v; synth -top mac8; stat'

layout:
	./bin/layout
