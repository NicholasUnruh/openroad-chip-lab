.PHONY: sim synth layout sim-training synth-training sim-training-netlist layout-training
.DEFAULT_GOAL := sim

HARDFLOAT = third_party/hardfloat
TRAINING_TOP ?= training_tile_v4
TRAINING_BUILD = build/$(subst _,-,$(subst training_tile,training,$(TRAINING_TOP)))
TRAINING_RTL = rtl/$(TRAINING_TOP).v $(if $(filter training_tile_v4,$(TRAINING_TOP)),rtl/training_tile_engine.v rtl/fp32_accum_fma_pipe.v) $(if $(filter training_tile_v6,$(TRAINING_TOP)),rtl/fp_dot8_fp32.v rtl/fp32_round.v) $(if $(filter training_tile_v7,$(TRAINING_TOP)),rtl/mx_dot32_fp32.v rtl/fp32_round.v) $(wildcard $(HARDFLOAT)/*.v) $(HARDFLOAT)/RISCV/HardFloat_specialize.v
TRAINING_INCLUDES = -I$(HARDFLOAT) -I$(HARDFLOAT)/RISCV
# Simulation model of the SRAM macro; synthesis reads it as a black box.
TRAINING_MACROS = $(if $(filter training_tile_v6 training_tile_v7,$(TRAINING_TOP)),tb/sky130_sram_1rw1r_128x256_8.v)
TRAINING_READ_MACROS = $(if $(TRAINING_MACROS),read_verilog -lib $(TRAINING_MACROS);)
TRAINING_TEST = tb/$(if $(filter training_tile_v6,$(TRAINING_TOP)),scratchpad_tile_test,$(if $(filter training_tile_v7,$(TRAINING_TOP)),mx_tile_test,training_tile_test)).cpp
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

.PHONY: sim-training-v7 synth-training-v7 sim-training-v7-netlist layout-training-v7 train-training-v7
sim-training-v7:
	$(MAKE) sim-training TRAINING_TOP=training_tile_v7
synth-training-v7:
	$(MAKE) synth-training TRAINING_TOP=training_tile_v7
sim-training-v7-netlist:
	$(MAKE) sim-training-netlist TRAINING_TOP=training_tile_v7
layout-training-v7:
	$(MAKE) layout-training TRAINING_TOP=training_tile_v7
# The study's MXFP8 training run with every GEMM on the Verilated V7 tile (about 30 min).
train-training-v7:
	verilator --cc --exe --build -j 4 --top-module training_tile_v7 --Mdir build/training-v7-train \
		rtl/training_tile_v7.v rtl/mx_dot32_fp32.v rtl/fp32_round.v tb/sky130_sram_1rw1r_128x256_8.v \
		$(CURDIR)/tb/mx_tile_train.cpp -CFLAGS '$(TRAINING_CFLAGS)'
	mkdir -p build/numerics
	./build/training-v7-train/Vtraining_tile_v7 3000 3 > build/numerics/train_mxfp8_v7_tile.txt
	grep '^final' build/numerics/train_mxfp8_v7_tile.txt

# ASAP7 PPA harnesses: flatten with Yosys, then run ORFS to route (bin/ppa).
# ppa_dot8_bf16_f<N> is the BF16 dot8 harness with a FRAME of N bits.
.PHONY: ppa ppa-netlist ppa-all
PPA_TOP ?= ppa_dot8_bf16
PPA_FRAME = $(patsubst ppa_dot8_bf16_f%,%,$(filter ppa_dot8_bf16_f%,$(PPA_TOP)))
PPA_MODULE = $(if $(PPA_FRAME),ppa_dot8_bf16,$(PPA_TOP))
PPA_RTL = rtl/ppa/$(PPA_MODULE).v rtl/fp_dot8_fp32.v rtl/fp32_round.v rtl/int8_dot8_int32.v rtl/mx_dot32_fp32.v rtl/fp32_accum_fma_pipe.v $(wildcard $(HARDFLOAT)/*.v) $(HARDFLOAT)/RISCV/HardFloat_specialize.v
ppa: ppa-netlist
	./bin/ppa $(PPA_TOP)
ppa-netlist:
	mkdir -p build/ppa
	yosys -Q -T -l build/ppa/$(PPA_TOP)_synth.log -p 'read_verilog $(TRAINING_INCLUDES) $(PPA_RTL); $(if $(PPA_FRAME),chparam -set FRAME $(PPA_FRAME) ppa_dot8_bf16;) synth -top $(PPA_MODULE) -flatten; $(if $(PPA_FRAME),rename ppa_dot8_bf16 $(PPA_TOP);) check -assert; splitnets; write_verilog -noattr build/ppa/$(PPA_TOP).v'
# Every study design point: flatten, route in one Bazel build, then routed power.
PPA_HARNESSES = ppa_fma_bf16 ppa_dot8_bf16 ppa_dot8_bf16_f24 ppa_dot8_bf16_f28 ppa_dot8_bf16_f36 \
	ppa_dot8_bf16_f40 ppa_dot8_e4m3 ppa_dot8_e5m2 ppa_dot8_int8 ppa_dot32_mxfp8 ppa_dot32_mxfp4 \
	ppa_dot32_mxfp8_e5m2 ppa_dot32_mxfp6_e3m2 ppa_dot32_mxfp6_e2m3 ppa_dot32_nvfp4
ppa-all:
	for top in $(PPA_HARNESSES); do $(MAKE) ppa-netlist PPA_TOP=$$top || exit 1; done
	./bin/ppa $(PPA_HARNESSES)
	for top in $(PPA_HARNESSES); do tools/ppa_power.sh $$top || exit 1; done

.PHONY: sim-dot8 sim-dot8-format sim-int8-dot8 sim-mx sim-mx-elem lint-mx formal-mx
# Dot-product unit in each input format: 0 = BF16, 1 = OCP FP8 E4M3, 2 = OCP FP8 E5M2;
# then the BF16 unit at the other alignment-frame widths of the numerics study.
DOT8_FRAME ?= 32
sim-dot8:
	for format in 0 1 2; do $(MAKE) sim-dot8-format DOT8_FORMAT=$$format || exit 1; done
	for frame in 24 28 36 40; do $(MAKE) sim-dot8-format DOT8_FORMAT=0 DOT8_FRAME=$$frame || exit 1; done
sim-int8-dot8:
	verilator --cc --exe --build -j 4 --top-module int8_dot8_int32 --Mdir build/int8-dot8-sim \
		rtl/int8_dot8_int32.v $(CURDIR)/tb/int8_dot8_test.cpp -CFLAGS '-std=c++17 -O2'
	./build/int8-dot8-sim/Vint8_dot8_int32
# MX block dot product, by element: 0 = MXFP8 E4M3, 1 = MXFP4 E2M1, 2 = MXFP8 E5M2,
# 3 = MXFP6 E3M2, 4 = MXFP6 E2M3, 5 = NVFP4 (E2M1, E4M3 scale per 16 elements).
sim-mx:
	for elem in 0 1 2 3 4 5; do $(MAKE) sim-mx-elem MX_ELEM=$$elem || exit 1; done
# Verible style lint of the MX units and V7; the disabled rules are SystemVerilog idioms
# (always_comb, typed parameters, explicit function lifetimes) this Verilog-2005 code avoids.
lint-mx:
	verible-verilog-lint --rules_config=tools/verible.rules rtl/mx_dot32_fp32.v rtl/fp32_round.v \
		rtl/training_tile_v7.v rtl/ppa/ppa_dot32_*.v formal/mx_dot32_formal.v
# Bounded proofs of every MX/NVFP4 unit against formal/mx_dot32_formal.v (several hours).
formal-mx:
	formal/run.sh 4
sim-mx-elem:
	verilator --cc --exe --build -j 4 --top-module mx_dot32_fp32 -GELEM=$(MX_ELEM) \
		--Mdir build/mx-$(MX_ELEM)-sim rtl/mx_dot32_fp32.v rtl/fp32_round.v $(CURDIR)/tb/mx_dot32_test.cpp \
		-CFLAGS '-std=c++17 -O2 -fno-fast-math -DMX_ELEM=$(MX_ELEM)'
	./build/mx-$(MX_ELEM)-sim/Vmx_dot32_fp32
sim-dot8-format:
	mkdir -p build
	verilator --cc --exe --build -j 4 --top-module fp_dot8_fp32 -GFORMAT=$(DOT8_FORMAT) -GFRAME=$(DOT8_FRAME) \
		--Mdir build/dot8-$(DOT8_FORMAT)-f$(DOT8_FRAME)-sim rtl/fp_dot8_fp32.v rtl/fp32_round.v $(CURDIR)/tb/fp_dot8_test.cpp \
		-CFLAGS '-std=c++17 -O2 -fno-fast-math -DDOT8_FORMAT=$(DOT8_FORMAT) -DDOT8_FRAME=$(DOT8_FRAME)'
	./build/dot8-$(DOT8_FORMAT)-f$(DOT8_FRAME)-sim/Vfp_dot8_fp32

check-training-structure:
	mkdir -p build
	yosys -Q -T -l $(TRAINING_BUILD)_structure.log -p '$(TRAINING_READ_MACROS) read_verilog $(TRAINING_INCLUDES) $(TRAINING_RTL); hierarchy -top $(TRAINING_TOP); proc; flatten; opt; wreduce; opt_clean; write_json $(TRAINING_BUILD)_structure.json'
	python3 tb/check_bf16_structure.py $(TRAINING_BUILD)_structure.json $(TRAINING_TOP)

# Numerics study accuracy (M4): GEMM error against FP64 and the bit-exact training emulation,
# GEMM error beside five training recipes at a time (six processes).
.PHONY: numerics
NUMERICS_DEPS = tools/numerics/datapaths.h tools/numerics/train.h tb/dot8_model.h tb/mx_model.h
TRAIN_RECIPES = fp32 bf16_fma bf16_dot8 bf16_dot8_f24 bf16_dot8_f28 bf16_dot8_f36 bf16_dot8_f40 \
	fp8_tensor mxfp8 mxfp4_fwd_mxfp8_bwd mxfp4 mxfp8_e5m2_bwd mxfp6_e3m2 mxfp6_e2m3 mxfp4_sr nvfp4 nvfp4_sr
build/%: tools/numerics/%.cpp $(NUMERICS_DEPS)
	mkdir -p build
	g++ $(TRAINING_CFLAGS) -o $@ $<
numerics: build/gemm_error build/train_mlp
	mkdir -p build/numerics
	./build/gemm_error > build/numerics/gemm_error.txt & \
	printf '%s\n' $(TRAIN_RECIPES) | xargs -P 5 -I{} sh -c './build/train_mlp 3000 3 {} > build/numerics/train_{}.txt'; \
	wait
	cat build/numerics/train_*.txt | grep '^final'

# Cycle-level V4/V6 performance model: its BENCH lines must equal both RTL testbenches'
# clock for clock; the sweep varies lanes/units, buffers/C banks and host port width.
.PHONY: model-check model-sweep
build/tile_model: model/tile_model.cpp
	mkdir -p build
	g++ -std=c++17 -O2 -Wall -Wextra -Werror -o $@ $<
model-check: build/tile_model
	$(MAKE) --no-print-directory sim-training-v4 | grep '^BENCH' > build/model_rtl_bench.txt
	$(MAKE) --no-print-directory sim-training-v6 | grep '^BENCH' >> build/model_rtl_bench.txt
	./build/tile_model bench > build/model_bench.txt
	diff build/model_rtl_bench.txt build/model_bench.txt
	@echo "PASS: tile_model matches $$(wc -l < build/model_bench.txt) RTL BENCH lines"
model-sweep: build/tile_model
	./build/tile_model sweep
