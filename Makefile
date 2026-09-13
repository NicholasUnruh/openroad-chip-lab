.PHONY: sim synth layout sim-training synth-training sim-training-netlist layout-training

HARDFLOAT = third_party/hardfloat
TRAINING_TOP ?= training_tile
TRAINING_BUILD = build/training$(if $(filter training_tile_v2,$(TRAINING_TOP)),-v2)
TRAINING_RTL = rtl/$(TRAINING_TOP).v $(if $(filter training_tile_v2,$(TRAINING_TOP)),rtl/fp32_fma_pipe.v) $(wildcard $(HARDFLOAT)/*.v) $(HARDFLOAT)/RISCV/HardFloat_specialize.v
TRAINING_INCLUDES = -I$(HARDFLOAT) -I$(HARDFLOAT)/RISCV
TRAINING_CFLAGS = -std=c++17 -O2 -fno-fast-math -ffp-contract=off $(if $(filter training_tile_v2,$(TRAINING_TOP)),-DTILE_V2)

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
	verilator --cc --exe --build -j 4 --top-module $(TRAINING_TOP) \
		--Mdir $(TRAINING_BUILD)-sim $(TRAINING_INCLUDES) \
		tb/hardfloat.vlt $(TRAINING_RTL) $(CURDIR)/tb/training_tile_test.cpp \
		-CFLAGS '$(TRAINING_CFLAGS)'
	./$(TRAINING_BUILD)-sim/V$(TRAINING_TOP)

synth-training:
	mkdir -p build
	yosys -Q -T -l $(TRAINING_BUILD)_synth.log -p 'read_verilog $(TRAINING_INCLUDES) $(TRAINING_RTL); synth -top $(TRAINING_TOP) -flatten; check -assert; stat; splitnets; write_verilog -noattr build/$(TRAINING_TOP).v'

sim-training-netlist: synth-training
	verilator --cc --exe --build -j 4 --top-module $(TRAINING_TOP) \
		--Mdir $(TRAINING_BUILD)-netlist-sim build/$(TRAINING_TOP).v \
		$(CURDIR)/tb/training_tile_test.cpp \
		-CFLAGS '$(TRAINING_CFLAGS)'
	./$(TRAINING_BUILD)-netlist-sim/V$(TRAINING_TOP)

layout-training: synth-training
	./bin/layout $(TRAINING_TOP)

.PHONY: sim-training-v2 synth-training-v2 sim-training-v2-netlist layout-training-v2
sim-training-v2:
	$(MAKE) sim-training TRAINING_TOP=training_tile_v2
synth-training-v2:
	$(MAKE) synth-training TRAINING_TOP=training_tile_v2
sim-training-v2-netlist:
	$(MAKE) sim-training-netlist TRAINING_TOP=training_tile_v2
layout-training-v2:
	$(MAKE) layout-training TRAINING_TOP=training_tile_v2
