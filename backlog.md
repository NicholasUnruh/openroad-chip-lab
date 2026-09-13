# Backlog

## In flight
## Queued
## Done
- [x] training_accelerator_fp32 - Build and verify an FP32 training tile in OpenROAD (repo: openroad-chip-lab) (done 2026-09-12)
  Completed FP32 4x4 training compute prototype. RTL and generic netlist each pass 402 jobs, including finite-difference gradients and 32 SGD updates. SKY130HD layout meets 100 ns constraints with +63.2336 ns setup and +0.467235 ns hold slack; zero router/antenna violations; GDS parsed by KLayout. Evidence and limitations: docs/training-results.md; interface: docs/training-tile.md. One shared FMA and host-managed training remain the baseline; pipelining, parallel lanes, SRAM and full-model execution are later milestones. No Git remote configured.
  Verified results: docs/training-results.md; implementation and documentation committed.
