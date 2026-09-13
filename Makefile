.PHONY: sim synth layout sim-training synth-training sim-training-netlist layout-training
.DEFAULT_GOAL := sim

HARDFLOAT = third_party/hardfloat
TRAINING_TOP ?= training_tile_v4
TRAINING_BUILD = build/$(subst _,-,$(subst training_tile,training,$(TRAINING_TOP)))
TRAINING_RTL = rtl/$(TRAINING_TOP).v $(if $(filter training_tile_v2 training_tile_v3 training_tile_v4,$(TRAINING_TOP)),rtl/training_tile_engine.v rtl/fp32_accum_fma_pipe.v) $(wildcard $(HARDFLOAT)/*.v) $(HARDFLOAT)/RISCV/HardFloat_specialize.v
TRAINING_INCLUDES = -I$(HARDFLOAT) -I$(HARDFLOAT)/RISCV
TRAINING_CFLAGS = -std=c++17 -O2 -fno-fast-math -ffp-contract=off $(if $(filter training_tile_v2,$(TRAINING_TOP)),-DTILE_V2) $(if $(filter training_tile_v3,$(TRAINING_TOP)),-DTILE_V3) $(if $(filter training_tile_v4,$(TRAINING_TOP)),-DTILE_V4)

ifeq ($(TRAINING_TOP),training_tile_v3)
sim-training synth-training: check-training-v3-structure
endif
ifeq ($(TRAINING_TOP),training_tile_v4)
sim-training synth-training: check-training-v4-structure
endif

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

.PHONY: sim-training-v3 synth-training-v3 sim-training-v3-netlist layout-training-v3 check-training-v3-structure
sim-training-v3:
	$(MAKE) sim-training TRAINING_TOP=training_tile_v3
synth-training-v3:
	$(MAKE) synth-training TRAINING_TOP=training_tile_v3
sim-training-v3-netlist:
	$(MAKE) sim-training-netlist TRAINING_TOP=training_tile_v3
layout-training-v3:
	$(MAKE) layout-training TRAINING_TOP=training_tile_v3

check-training-v3-structure:
	$(MAKE) check-training-structure TRAINING_TOP=training_tile_v3

.PHONY: sim-training-v4 synth-training-v4 sim-training-v4-netlist layout-training-v4 check-training-v4-structure check-training-structure sim-training-v4-walkthrough
sim-training-v4:
	$(MAKE) sim-training TRAINING_TOP=training_tile_v4
sim-training-v4-walkthrough: TRAINING_TOP = training_tile_v4
sim-training-v4-walkthrough:
	verilator --binary --timing --trace --timescale 1ns/1ps -j 4 --top-module training_tile_v4_walkthrough \
		--Mdir build/training-v4-walkthrough-sim $(TRAINING_INCLUDES) \
		tb/hardfloat.vlt $(TRAINING_RTL) tb/training_tile_v4_walkthrough.sv
	./build/training-v4-walkthrough-sim/Vtraining_tile_v4_walkthrough
synth-training-v4:
	$(MAKE) synth-training TRAINING_TOP=training_tile_v4
sim-training-v4-netlist:
	$(MAKE) sim-training-netlist TRAINING_TOP=training_tile_v4
layout-training-v4:
	$(MAKE) layout-training TRAINING_TOP=training_tile_v4
check-training-v4-structure:
	$(MAKE) check-training-structure TRAINING_TOP=training_tile_v4

check-training-structure:
	mkdir -p build
	yosys -Q -T -l $(TRAINING_BUILD)_structure.log -p 'read_verilog $(TRAINING_INCLUDES) $(TRAINING_RTL); hierarchy -top $(TRAINING_TOP); proc; flatten; opt; wreduce; opt_clean; write_json $(TRAINING_BUILD)_structure.json'
	python3 tb/check_bf16_structure.py $(TRAINING_BUILD)_structure.json $(TRAINING_TOP)
