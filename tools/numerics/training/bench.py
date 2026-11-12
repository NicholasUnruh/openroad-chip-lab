"""Throughput of the emulated GEMMs at GPT-2 small shapes against cuBLAS BF16 and FP32.

For each format it times the fused quantizer (kernel.pack) and the block GEMM for the forward GEMM of each block Linear with
T = 8192 tokens, and reports effective TFLOP/s (2 M N K / time).

usage: uv run --extra cuda python bench.py [--out results.txt]
"""

import argparse

import torch

import kernel
import mx

SHAPES = {  # name: (M, N, K) for C = A B^T, A = activations [T, in], B = weight [out, in]
    "qkv": (8192, 2304, 768),
    "proj": (8192, 768, 768),
    "fc": (8192, 3072, 768),
    "mlp_proj": (8192, 768, 3072),
}


def timed(fn, repeats: int = 10) -> float:
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats / 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    lines = [f"{'format':12s} {'gemm':9s} {'quant ms':>9s} {'gemm ms':>9s} {'TFLOP/s':>8s}"]
    for name, (m, n, k) in SHAPES.items():
        a = torch.randn(m, k, device="cuda")
        b = torch.randn(n, k, device="cuda") * 0.02
        flops = 2 * m * n * k
        for label, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
            x, y = a.to(dtype), b.to(dtype)
            t = timed(lambda: x @ y.T)
            lines.append(f"{label:12s} {name:9s} {0:9.2f} {t * 1e3:9.2f} {flops / t / 1e12:8.2f}")
        for fmt in (mx.MXFP8, mx.MXFP6_E3M2, mx.MXFP6_E2M3, mx.MXFP4, mx.NVFP4):
            tq = timed(lambda: (kernel.pack(a, fmt), kernel.pack(b, fmt)))
            pa, pb = kernel.pack(a, fmt), kernel.pack(b, fmt)
            tg = timed(lambda: kernel.gemm_packed(pa, pb))
            total = tq + tg
            lines.append(
                f"{fmt.name:12s} {name:9s} {tq * 1e3:9.2f} {tg * 1e3:9.2f} {flops / total / 1e12:8.2f}"
            )
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
