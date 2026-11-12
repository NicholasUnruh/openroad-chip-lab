# Numerics study results

Method: [numerics-study.md](numerics-study.md). Verified 2026-11-11. **ASAP7 is a predictive PDK: every figure below is only a comparison between design points, never a foundry number.**

## Answer

Among datapaths that train within the tolerance, **MXFP8 32-term block dot products** give the most throughput per mm² (2.17× the BF16 dot8) and per joule (2.33×). Per-tensor FP8 dot8 units come next at about 1.5–1.6×. **MXFP4 is 5.9× denser but fails training** (+8.9% test loss). For BF16, the frame width is a free knob: narrowing the alignment frame from 32 to 24 bits saves 11% area and 8% energy, and lifts fmax above 1 GHz, with no measurable change in error or training. Chained FMAs cost 3.3× the dot8's energy per MAC and give 0.34× its throughput per mm².

## Ranking

Throughput per mm² uses the routed fmax; energy uses simulated activity at 1 GHz. Rank among the points that pass the training gate.

| Rank | Point | Trains | GMAC/s/mm² | × dot8_bf16 | MAC/nJ | × dot8_bf16 |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `dot32_mxfp8` | yes | 11,578 | 2.17 | 2,406 | 2.33 |
| 2 | `dot8_e4m3` | yes (FP8 recipe) | 8,783 | 1.64 | 1,509 | 1.46 |
| 3 | `dot8_e5m2` | yes (FP8 recipe) | 8,287 | 1.55 | 1,566 | 1.51 |
| 4 | `dot8_bf16_f24` | yes | 6,930 | 1.30 | 1,125 | 1.09 |
| 5 | `dot8_bf16_f28` | yes | 6,629 | 1.24 | 1,074 | 1.04 |
| 6 | `dot8_bf16` (frame 32, V6 unit) | yes | 5,347 | 1.00 | 1,034 | 1.00 |
| 7 | `dot8_bf16_f36` | yes | 5,137 | 0.96 | 1,001 | 0.97 |
| 8 | `dot8_bf16_f40` | yes | 5,097 | 0.95 | 979 | 0.95 |
| 9 | `fma_bf16` | yes | 1,800 | 0.34 | 313 | 0.30 |
| — | `dot32_mxfp4` | no | 31,348 | 5.86 | 4,992 | 4.83 |
| — | `dot8_int8` | inference only | 26,476 | 4.95 | 2,974 | 2.88 |

![Throughput per mm² and MACs per nJ against final test loss relative to FP32, with the 1% gate](img/numerics-pareto.png)

*Each datapath's density and energy against its training loss; only points left of the 1% gate count. MXFP8 is the best point that passes; MXFP4 is far denser but lands at +8.9%.*

E4M3 and E5M2 are one recipe (E4M3 forward, E5M2 gradients), so a training chip needs both or a unit that handles both.

## Area, fmax and energy

Each unit is routed behind registered inputs in a fixed harness, with a 1 GHz target. Area is standard-cell area, including the harness registers.

| Point | MACs/clk | Area µm² | Area/MAC µm² | fmax MHz | Power at 1 GHz mW | Energy fJ/MAC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fma_bf16` | 1 | 579 | 579 | 1042 | 3.19 | 3190 |
| `dot8_bf16` | 8 | 1319 | 165 | 882 | 7.74 | 968 |
| `dot8_bf16_f24` | 8 | 1170 | 146 | 1014 | 7.11 | 889 |
| `dot8_bf16_f28` | 8 | 1236 | 155 | 1024 | 7.45 | 931 |
| `dot8_bf16_f36` | 8 | 1402 | 175 | 900 | 7.99 | 999 |
| `dot8_bf16_f40` | 8 | 1448 | 181 | 923 | 8.17 | 1021 |
| `dot8_e4m3` | 8 | 938 | 117 | 1030 | 5.30 | 662 |
| `dot8_e5m2` | 8 | 932 | 116 | 965 | 5.11 | 639 |
| `dot32_mxfp8` | 32 | 2753 | 86 | 996 | 13.30 | 416 |
| `dot32_mxfp4` | 32 | 1053 | 33 | 1032 | 6.41 | 200 |
| `dot8_int8` | 8 | 435 | 54 | 1441 | 2.69 | 336 |

All 11 routes pass hold, slew, capacitance, fanout and router DRC. Setup slack is the fmax measurement, not a gate. fmax does not change monotonically with frame width (32: 882, 36: 900, 40: 923 MHz), and a functionally identical earlier netlist of each `fp_dot8` point routed up to 4% different in fmax and area, so differences under about 5% between neighbouring points are placement and repair noise.

**Energy method.** Each routed netlist is simulated with zero-delay cell models generated from the ASAP7 Liberty functions. Stimulus is 2,000 clocks of N(0, 1) operands quantized to the point's format, with N(0, 16) FP32 accumulator inputs. The routed outputs must match the Yosys netlist bit for bit after warm-up, and they do for all 11 points. OpenSTA reads the VCD (100% of pins annotated) and extracted SPEF, then reports power. The earlier probabilistic input-activity method put `fma_bf16` at 17.7 mW, 5.5× the simulated value, because it propagates toggles through the deep arithmetic logic without correlation. Zero-delay simulation leaves out glitch power, so absolute energies are low by an unknown factor; the comparison assumes it is similar across points.

## GEMM error

Median / p99 of |y − y₆₄| / Σ|aᵢbᵢ| over 2,000 dot products per cell. Every BF16 variant matches `bf16_fma` to three digits at every frame width: BF16 input rounding (relative 2⁻⁹) swamps the frame truncation, which even at 24 bits sits 16 bits below the largest term.

| Datapath | Gaussian K=4096 | Laplace K=4096 | Outliers K=4096 | Row-scaled K=4096 |
| --- | ---: | ---: | ---: | ---: |
| `bf16_fma`, `bf16_dot8` (all frames) | 3.6e-5 / 1.4e-4 | 4.7e-5 / 1.9e-4 | 1.9e-4 / 7.8e-4 | 3.9e-5 / 1.5e-4 |
| `e4m3_tensor` | 6.2e-4 / 2.3e-3 | 8.2e-4 / 3.1e-3 | 3.0e-3 / 1.1e-2 | 6.2e-4 / 2.4e-3 |
| `e5m2_tensor` | 1.2e-3 / 4.6e-3 | 1.6e-3 / 6.0e-3 | 6.0e-3 / 2.4e-2 | 1.2e-3 / 4.7e-3 |
| `mxfp8` | 7.0e-4 / 2.6e-3 | 9.8e-4 / 3.7e-3 | 3.8e-3 / 1.6e-2 | 7.1e-4 / 2.8e-3 |
| `mxfp4` | 2.6e-3 / 1.1e-2 | 3.8e-3 / 1.5e-2 | 1.4e-2 / 5.7e-2 | 2.6e-3 / 9.7e-3 |
| `int8_tensor` | 2.8e-4 / 1.0e-3 | 7.7e-4 / 2.8e-3 | 7.2e-3 / 2.9e-2 | 1.3e-2 / 5.9e-2 |
| `int8_row` | 2.1e-4 / 7.9e-4 | 4.3e-4 / 1.6e-3 | 4.8e-3 / 2.0e-2 | 2.0e-4 / 7.9e-4 |

K=256 behaves the same way; the full table is in `results/numerics/gemm_error.txt`. Per-tensor INT8 collapses on row-scaled data (activations whose magnitude varies by token); MX block scaling does not.

## Training emulation

A 64-64-10 ReLU student learns a fixed random teacher. Every forward, dW and dX GEMM is emulated bit-exactly through the datapath, with FP32 master weights and host SGD. Each step draws a fresh batch of 64; the learning rate decays linearly from 0.2 to 0.02; the run is 3,000 steps with 3 seeds. Each point is scored by its final test loss on 2,048 held-out samples.

| Recipe (forward / backward) | Test loss | Seed std | Accuracy | Within 1% of FP32 |
| --- | ---: | ---: | ---: | --- |
| FP32 | 0.8145 | 0.0027 | 0.698 | — |
| BF16 FMA | 0.8146 | 0.0024 | 0.697 | yes |
| BF16 dot8, frames 24–40 | 0.8147–0.8149 | ≤0.0029 | 0.697–0.699 | yes |
| FP8 per-tensor (E4M3 / E5M2) | 0.8214 | 0.0055 | 0.694 | yes (+0.85%) |
| MXFP8 | 0.8176 | 0.0051 | 0.696 | yes (+0.38%) |
| MXFP4 / MXFP8 | 0.8860 | 0.0091 | 0.672 | no (+8.8%) |
| MXFP4 | 0.8867 | 0.0094 | 0.669 | no (+8.9%) |

Loss curves (10 points per run) are in `results/numerics/train_*.txt`; `tools/plot_figures.py` draws them below.

![Test loss during training for FP32, BF16 dot8, FP8, MXFP8 and MXFP4](img/numerics-loss-curves.png)

*Mean test loss of three seeds: every format except MXFP4 tracks FP32 to the end.* MXFP4 fails even with MXFP8 gradients, so the forward pass is where FP4 loses accuracy.

## Choices and limitations

- **Tolerance (gate).** The gate is 1% of FP32's mean final test loss, about 3× FP32's seed spread. Per-tensor FP8 passes at +0.85%, so it is marginal: a 0.75% gate would fail it, while MXFP8 passes either way.
- **Fresh batches.** Reusing a fixed 4,096-sample set for about 47 epochs overfits (train loss 0.06 against test loss 2.3), so overfitting rather than arithmetic would decide the ranking. Each step therefore draws a fresh batch and the learning rate decays.
- **Optimistic FP8 scaling.** Per-tensor FP8 scales each tensor from its own current amax (just-in-time). Delayed scaling, which production recipes use, is likely worse.
- **Small model.** A 64-unit MLP is far from transformer training. The training result only orders the formats and should not be read as a prediction at scale.
- **Accumulator input.** The harness drives an independent random accumulator every clock instead of feeding the result back, so accumulator toggle rates are approximate.
- **Frame-width variants** use the BF16 unit's new `FRAME` parameter. Every width passes 4.2M vectors bit-exact against the specification and the exact-sum oracle (`make sim-dot8`).

## Reproduction

```sh
make sim-dot8 sim-mx sim-int8-dot8   # unit correctness, all formats and frames
make ppa-all                          # 11 routes (about 30 min at ORFS_JOBS=6), then power
make numerics                         # GEMM error and training emulation (about 20 min)
python3 tools/numerics/report.py      # every table above
```
