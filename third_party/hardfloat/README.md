# Berkeley HardFloat subset

Unmodified Verilog sources from John Hauser's [HardFloat Release 1](https://www.jhauser.us/arithmetic/HardFloat.html), downloaded 2026-09-12. Only the FP32 fused multiply-add dependencies and RISC-V specialization are included.

Archive: [HardFloat-1.zip](https://www.jhauser.us/arithmetic/HardFloat-1.zip)

Archive SHA256: `6b3757c9fbfa2230c6a2b84605e39372cb589dd7500e979c4f0b8ecc8a03b14b`.

Each source retains the upstream BSD license and original bytes. Verify them with:

```sh
cd third_party/hardfloat
sha256sum -c SHA256SUMS
```

The lab fixes round-to-nearest-even and tininess detection after rounding. RISC-V specialization canonicalizes NaNs. See [the module documentation](https://www.jhauser.us/arithmetic/HardFloat-1/doc/HardFloat-Verilog.html) and [the training tile interface](../../docs/training-v4-interface.md).

The Verilator configuration in `tb/hardfloat.vlt` suppresses implicit-width diagnostics only inside this unmodified dependency; lab RTL diagnostics remain enabled.
