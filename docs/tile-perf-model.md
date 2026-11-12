# Tile performance model

`model/tile_model.cpp` is a cycle-level model of `training_tile_v4` and `training_tile_v6`. It models
timing only, not arithmetic. It reproduces every BENCH line of both RTL testbenches clock for clock,
then sweeps the parameters the RTL fixes.

```sh
make model-check   # runs sim-training-v4 and sim-training-v6, diffs the 9 BENCH lines exactly
make model-sweep   # prints the tables below
```

## What is modelled

| Tile | Model | Source of truth |
| --- | --- | --- |
| V4 | `512/lanes + 3` busy clocks per 8×8×8 job, done on the last. A start beats a simultaneous write. Writes only to a buffer the tile is not computing on. One write and one read transfer per clock. The host is greedy: in-order writes as soon as a buffer frees, reads as soon as a job is done. | `tb/bf16_tile_tests.h`, [V4 interface](training-v4-interface.md) |
| V6 | Sequencer control transliterated register for register: loader (16 scratchpad rows per job), waiting job, engine (`64/units` clocks per job), 6-stage dot-product pipeline, C-bank drain FIFO. The host loop matches `tb/scratchpad_tile_test.cpp`. | `rtl/training_tile_v6.v` |

Matched at the RTL point (lanes 8, two buffers, 4-byte ports; four units, two C banks), clocks:

| Workload | V4 | V6 |
| --- | --- | --- |
| 64 independent 8×8 jobs, serial / overlapped | 12,544 / 4,480 | — |
| 64 accumulate jobs (C loaded), serial / overlapped | 16,640 / 8,387 | — |
| GEMM 8³ ×64, 16³ ×8, 32³ ×1 streamed | 4,480 each | — |
| 32³ GEMM incl. scratchpad fill | — | 2,181 |
| Eight 16³ GEMMs, next one loading during the current one | — | 2,720 |

The V6 32³ figure is 2,181 on current RTL; the 2,165 quoted in older notes predates the dot8 pipeline stages.

## V4: lanes × host port (32³ streamed GEMM, clocks)

Two buffers. A third buffer gives the same count at every point; one buffer is shown for 8 lanes.

| Lanes | 2 B/clk | 4 B/clk | 8 B/clk | 16 B/clk |
| --- | --- | --- | --- | --- |
| 4 | 8,704 | 8,576 | 8,512 | 8,480 |
| 8 (RTL) | 8,451 | **4,480** | 4,416 | 4,384 |
| 8, one buffer | 12,672 | 8,512 | 6,432 | 5,392 |
| 16 | 8,419 | 4,259 | 2,368 | 2,336 |
| 32 | 8,403 | 4,243 | 2,163 | 1,312 |
| 64 | 8,395 | 4,235 | 2,155 | 1,115 |

The steady-state period per job is `1 + max(512/lanes + 3, 256/port)`: compute against operand writes,
plus the start clock that carries no write. The 64 jobs add a fill and a drain.

- **The RTL point is balanced:** 67 compute clocks against 64 write clocks. More lanes at 4 B/clk buy 5%
  (4,480 → 4,259): the write port bounds V4.
- **Lanes and port scale together:** 16 lanes need 8 B/clk (1.89×), 32 lanes need 16 B/clk (3.41×).
  64 lanes at 16 B/clk are write-bound again (16 write vs 11 compute clocks): 4.02×.
- **Double buffering is the overlap win** (8,512 → 4,480 at the RTL point). A third buffer buys nothing:
  a fixed-order stream has no jitter for it to absorb.

## V6: dot units × host port (clocks)

Two C banks.

| Units (MACs/clk) | 32³ at 2 / 4 / 8 / 16 B/clk | Eight 16³ at 2 / 4 / 8 / 16 B/clk |
| --- | --- | --- |
| 2 (16) | 4,293 / 3,156 / 2,612 / 2,340 | 5,280 / 3,032 / 2,592 / 2,400 |
| 4 (32, RTL) | 4,180 / **2,181** / 1,588 / 1,316 | 5,024 / **2,720** / 1,624 / 1,376 |
| 8 (64) | 4,172 / 2,124 / 1,580 / 1,308 | 4,960 / 2,656 / 1,504 / 1,312 |

C banks, four units:

| C banks | 32³ at 2 / 4 / 8 / 16 B/clk | Eight 16³ at 2 / 4 / 8 / 16 B/clk |
| --- | --- | --- |
| 1 | 5,245 / 3,197 / 2,173 / 1,661 | 5,960 / 3,656 / 2,504 / 1,928 |
| 2 (RTL) | 4,180 / 2,181 / 1,588 / 1,316 | 5,024 / 2,720 / 1,624 / 1,376 |
| 3 | 4,180 / 2,132 / 1,588 / 1,316 | 5,024 / 2,720 / 1,568 / 1,376 |

- **V6 is host-bound at 4 B/clk.** In the 32³ run, 1,024 of the 2,181 clocks are the 4 KB scratchpad
  fill. Each 16³ GEMM moves 1 KB in and 1 KB out (256 clocks per port) against 128 compute clocks.
  A 16 B/clk port gives 1.66× (32³) and 1.98× (16³ stream).
- **More dot units do not pay.** Eight units save at most 8%: the loader feeds one job per 16 clocks
  (16 scratchpad rows), which equals the four-unit run length. After the port, the next lever is
  scratchpad read width, not more units.
- **Two C banks are right.** One bank serializes each output drain against the next tile (+47% at the
  RTL point). A third saves 2–3% at two points and nothing elsewhere.

## Limits

- **Wider ports are host-side only.** They assume the tile's banks widen to match; the RTL ports are
  32-bit.
- **V4 bank selects are modelled independently.** The RTL ties the read and write bank selects to one
  `host_bank`. With up to two buffers this is identical, because the buffer being read is the next one
  written. Three buffers would need separate selects.
- **V6 units change only the run length** (`64/units` clocks per job); the loader stays at one row per
  clock. One C bank needs an in-flight guard the RTL does not have, which the model adds: with one bank,
  a `k = 0` job could otherwise clear C while the previous tile's last job is still in the pipeline.
  With two banks the guard never fires.
- **Clocks only, at a fixed clock.** Area and fmax for these points need synthesis and layout;
  see [V4 results](training-v4-results.md) and [V6 results](training-v6-results.md).
