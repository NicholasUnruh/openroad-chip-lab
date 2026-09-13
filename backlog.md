# Backlog

## In flight
## Queued
## Done
- [x] training_tile_v3 - Build native BF16 accelerator with buffered transfers and FP32 accumulation (repo: openroad-chip-lab) (done 2026-09-12)
  Complete: native BF16 implementation ba50e4c and physical repair/gate 0d046b5. RTL/netlist each pass 4868 functional/training jobs plus 256 benchmark tiles; all 65536 BF16 A encodings exercised. Four 8x8 multipliers; BF16 operand banks, FP32 accumulation. Buffered stream uses 1312 vs V2 4352 clocks for 64 tiles (3.32x at equal clock). Revised 40 MHz layout passes all timing/electrical/router gates and antenna checks: setup +10.0163 ns, hold +0.405551 ns, area 251834 um2 (13.39% below recorded V2). KLayout reads GDS. Evidence and limitations: docs/training-v3-results.md; interface: docs/training-v3.md. Local commits only; no Git remote configured.
- [x] training_tile_v2 - Build and measure a pipelined four-lane FP32 training tile (repo: openroad-chip-lab) (done 2026-09-12)
  Complete: implementation 6b5e8dc. Both scalar/v2 RTL and generic netlists pass 405 functional/training jobs plus 256 timed tiles each. V2 routes at 40 MHz with setup +10.0459 ns, hold +0.407863 ns; zero routing, antenna and electrical violations. Estimated fmax 66.8711 MHz; standard-cell area 290778 um2. Streamed projection 75.29 MFLOPS at target, 4.09x scalar at respective estimated maxima for 2.33x cell area. KLayout reads GDS; MAC8 and scalar cached layout regressions pass. Evidence and reproduction: docs/training-v2-results.md. Power and fabrication signoff remain unqualified.
  Implemented, numerically verified, routed and documented; local commit only because repository has no remote.
- [x] training_accelerator_fp32 - Build and verify an FP32 training tile in OpenROAD (repo: openroad-chip-lab) (done 2026-09-12)
  Completed FP32 4x4 training compute prototype. RTL and generic netlist each pass 402 jobs, including finite-difference gradients and 32 SGD updates. SKY130HD layout meets 100 ns constraints with +63.2336 ns setup and +0.467235 ns hold slack; zero router/antenna violations; GDS parsed by KLayout. Evidence and limitations: docs/training-results.md; interface: docs/training-tile.md. One shared FMA and host-managed training remain the baseline; pipelining, parallel lanes, SRAM and full-model execution are later milestones. No Git remote configured.
  Verified results: docs/training-results.md; implementation and documentation committed.
