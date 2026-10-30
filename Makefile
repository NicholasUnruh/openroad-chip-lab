.PHONY: sim synth layout sim-training synth-training sim-training-netlist layout-training
.DEFAULT_GOAL := sim

HARDFLOAT = third_party/hardfloat
TRAINING_TOP ?= training_tile_v4
TRAINING_BUILD = build/$(subst _,-,$(subst training_tile,training,$(TRAINING_TOP)))
TRAINING_RTL = rtl/$(TRAINING_TOP).v $(if $(filter training_tile_v4,$(TRAINING_TOP)),rtl/training_tile_engine.v rtl/fp32_accum_fma_pipe.v) $(if $(filter training_tile_v6,$(TRAINING_TOP)),rtl/fp_dot8_fp32.v rtl/fp32_round.v) $(wildcard $(HARDFLOAT)/*.v) $(HARDFLOAT)/RISCV/HardFloat_specialize.v
TRAINING_INCLUDES = -I$(HARDFLOAT) -I$(HARDFLOAT)/RISCV
# Simulation model of the SRAM macro; synthesis reads it as a black box.
TRAINING_MACROS = $(if $(filter training_tile_v6,$(TRAINING_TOP)),tb/sky130_sram_1rw1r_128x256_8.v)
TRAINING_READ_MACROS = $(if $(TRAINING_MACROS),read_verilog -lib $(TRAINING_MACROS);)
TRAINING_TEST = tb/$(if $(filter training_tile_v6,$(TRAINING_TOP)),scratchpad_tile_test,training_tile_test).cpp
TRAINING_CFLAGS = -std=c++17 -O2 -fno-fast-math -ffp-contract=off

ifeq ($(TRAINING_TOP),training_tile_v4)
sim-training synth-training: check-training-v4-structure
endif
ifeq ($(TRAINING_TOP),training_tile_v6)
sim-training synth-training: check-training-v6-structure
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
		tb/hardfloat.vlt $(TRAINING_RTL) $(TRAINING_MACROS) $(CURDIR)/$(TRAINING_TEST) \
		-CFLAGS '$(TRAINING_CFLAGS)'
	./$(TRAINING_BUILD)-sim/V$(TRAINING_TOP)

synth-training:
	mkdir -p build
	yosys -Q -T -l $(TRAINING_BUILD)_synth.log -p '$(TRAINING_READ_MACROS) read_verilog $(TRAINING_INCLUDES) $(TRAINING_RTL); synth -top $(TRAINING_TOP) -flatten; check -assert; stat; splitnets; write_verilog -noattr build/$(TRAINING_TOP).v'

sim-training-netlist: synth-training
	verilator --cc --exe --build -j 4 --top-module $(TRAINING_TOP) \
		--Mdir $(TRAINING_BUILD)-netlist-sim build/$(TRAINING_TOP).v $(TRAINING_MACROS) \
		$(CURDIR)/$(TRAINING_TEST) \
		-CFLAGS '$(TRAINING_CFLAGS)'
	./$(TRAINING_BUILD)-netlist-sim/V$(TRAINING_TOP)

layout-training: synth-training
	./bin/layout $(TRAINING_TOP)

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

.PHONY: sim-training-v6 synth-training-v6 sim-training-v6-netlist layout-training-v6 check-training-v6-structure
sim-training-v6:
	$(MAKE) sim-training TRAINING_TOP=training_tile_v6
synth-training-v6:
	$(MAKE) synth-training TRAINING_TOP=training_tile_v6
sim-training-v6-netlist:
	$(MAKE) sim-training-netlist TRAINING_TOP=training_tile_v6
layout-training-v6:
	$(MAKE) layout-training TRAINING_TOP=training_tile_v6
check-training-v6-structure:
	$(MAKE) check-training-structure TRAINING_TOP=training_tile_v6

# ASAP7 PPA harnesses: flatten with Yosys, then run ORFS to route (bin/ppa).
.PHONY: ppa
PPA_TOP ?= ppa_dot8_bf16
PPA_RTL = rtl/ppa/$(PPA_TOP).v rtl/fp_dot8_fp32.v rtl/fp32_round.v rtl/int8_dot8_int32.v rtl/mx_dot32_fp32.v rtl/fp32_accum_fma_pipe.v $(wildcard $(HARDFLOAT)/*.v) $(HARDFLOAT)/RISCV/HardFloat_specialize.v
ppa:
	mkdir -p build/ppa
	yosys -Q -T -l build/ppa/$(PPA_TOP)_synth.log -p 'read_verilog $(TRAINING_INCLUDES) $(PPA_RTL); synth -top $(PPA_TOP) -flatten; check -assert; splitnets; write_verilog -noattr build/ppa/$(PPA_TOP).v'
	./bin/ppa $(PPA_TOP)

.PHONY: sim-dot8 sim-dot8-format sim-int8-dot8 sim-mx sim-mx-elem
# Dot-product unit in each input format: 0 = BF16, 1 = OCP FP8 E4M3, 2 = OCP FP8 E5M2.
sim-dot8:
	for format in 0 1 2; do $(MAKE) sim-dot8-format DOT8_FORMAT=$$format || exit 1; done
sim-int8-dot8:
	verilator --cc --exe --build -j 4 --top-module int8_dot8_int32 --Mdir build/int8-dot8-sim \
		rtl/int8_dot8_int32.v $(CURDIR)/tb/int8_dot8_test.cpp -CFLAGS '-std=c++17 -O2'
	./build/int8-dot8-sim/Vint8_dot8_int32
# MX block dot product: 0 = MXFP8 (E4M3 elements), 1 = MXFP4 (E2M1 elements).
sim-mx:
	for elem in 0 1; do $(MAKE) sim-mx-elem MX_ELEM=$$elem || exit 1; done
sim-mx-elem:
	verilator --cc --exe --build -j 4 --top-module mx_dot32_fp32 -GELEM=$(MX_ELEM) \
		--Mdir build/mx-$(MX_ELEM)-sim rtl/mx_dot32_fp32.v rtl/fp32_round.v $(CURDIR)/tb/mx_dot32_test.cpp \
		-CFLAGS '-std=c++17 -O2 -fno-fast-math -DMX_ELEM=$(MX_ELEM)'
	./build/mx-$(MX_ELEM)-sim/Vmx_dot32_fp32
sim-dot8-format:
	mkdir -p build
	verilator --cc --exe --build -j 4 --top-module fp_dot8_fp32 -GFORMAT=$(DOT8_FORMAT) \
		--Mdir build/dot8-$(DOT8_FORMAT)-sim rtl/fp_dot8_fp32.v rtl/fp32_round.v $(CURDIR)/tb/fp_dot8_test.cpp \
		-CFLAGS '-std=c++17 -O2 -fno-fast-math -DDOT8_FORMAT=$(DOT8_FORMAT)'
	./build/dot8-$(DOT8_FORMAT)-sim/Vfp_dot8_fp32

check-training-structure:
	mkdir -p build
	yosys -Q -T -l $(TRAINING_BUILD)_structure.log -p '$(TRAINING_READ_MACROS) read_verilog $(TRAINING_INCLUDES) $(TRAINING_RTL); hierarchy -top $(TRAINING_TOP); proc; flatten; opt; wreduce; opt_clean; write_json $(TRAINING_BUILD)_structure.json'
	python3 tb/check_bf16_structure.py $(TRAINING_BUILD)_structure.json $(TRAINING_TOP)
