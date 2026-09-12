.PHONY: sim synth layout sim-training synth-training sim-training-netlist layout-training

HARDFLOAT = third_party/hardfloat
TRAINING_RTL = rtl/training_tile.v $(wildcard $(HARDFLOAT)/*.v) $(HARDFLOAT)/RISCV/HardFloat_specialize.v
TRAINING_INCLUDES = -I$(HARDFLOAT) -I$(HARDFLOAT)/RISCV

sim:
	mkdir -p build
	iverilog -g2012 -s mac8_tb -o build/mac8_tb.vvp rtl/mac8.v tb/mac8_tb.v
	vvp build/mac8_tb.vvp

synth:
	yosys -Q -T -p 'read_verilog rtl/mac8.v; synth -top mac8; stat'

layout:
	./bin/layout

sim-training:
	mkdir -p build
	verilator --cc --exe --build -j 4 --top-module training_tile \
		--Mdir build/training-sim $(TRAINING_INCLUDES) \
		tb/hardfloat.vlt $(TRAINING_RTL) $(CURDIR)/tb/training_tile_test.cpp \
		-CFLAGS '-std=c++17 -O2 -fno-fast-math -ffp-contract=off'
	./build/training-sim/Vtraining_tile

synth-training:
	mkdir -p build
	yosys -Q -T -l build/training_synth.log -p 'read_verilog $(TRAINING_INCLUDES) $(TRAINING_RTL); synth -top training_tile -flatten; check -assert; stat; splitnets; write_verilog -noattr build/training_tile.v'

sim-training-netlist: synth-training
	verilator --cc --exe --build -j 4 --top-module training_tile \
		--Mdir build/training-netlist-sim build/training_tile.v \
		$(CURDIR)/tb/training_tile_test.cpp \
		-CFLAGS '-std=c++17 -O2 -fno-fast-math -ffp-contract=off'
	./build/training-netlist-sim/Vtraining_tile

layout-training: synth-training
	./bin/layout training_tile
