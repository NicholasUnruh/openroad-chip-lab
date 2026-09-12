# OpenROAD chip lab

- Scope: digital ASIC experiments using the SKY130HD process. Keep FPGA projects in their own repo.
- Verify RTL behavior with `make sim` and logic synthesis with `make synth`.
- Run `make layout` for the SKY130HD physical flow. It uses a pinned upstream bazel-orfs checkout and selects real KLayout for GDS.
- Inspect logs, timing, routing and physical checks before claiming success. A produced GDS alone is not signoff.
- A core layout is not a manufacturable chip. Pad frame, ESD, power, integration, DRC, LVS, timing signoff, and foundry acceptance require separate evidence.
- Keep generated files out of Git. Keep README at most 200 words and place detailed lessons under `docs/`.
