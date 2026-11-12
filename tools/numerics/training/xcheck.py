"""Randomized bit-exact cross-check of mx.py (and the GPU kernel) against the C++ references.

For every format and data distribution, the same FP32 matrices go through xcheck.cpp (the
numerics study's to_small, mx_chain and exact-sum oracle) and through mx.quantize /
mx.gemm_reference. Dequantized elements and GEMM outputs must agree bit for bit. With --kernel,
the CUDA kernel's output must agree too.

usage: uv run --extra cpu python xcheck.py [--cases N] [--kernel]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import mx

HERE = Path(__file__).resolve().parent
BINARY = HERE.parents[2] / "build" / "mx_xcheck"
FORMAT_IDS = ["mxfp8", "mxfp8_e5m2", "mxfp6_e3m2", "mxfp6_e2m3", "mxfp4", "nvfp4"]


def build() -> None:
    BINARY.parent.mkdir(exist_ok=True)
    cmd = ["g++", "-std=c++17", "-O2", "-fno-fast-math", "-ffp-contract=off"]
    subprocess.run([*cmd, "-o", str(BINARY), str(HERE / "xcheck.cpp")], check=True, timeout=300)


def sample(kind: str, shape: tuple[int, int], g: torch.Generator) -> torch.Tensor:
    """Test matrices that stress scaling, saturation, subnormals and rounding ties."""
    x = torch.randn(shape, generator=g, dtype=torch.float64)
    if kind == "laplace":
        u = torch.rand(shape, generator=g, dtype=torch.float64)
        x = -torch.log1p(-u) * torch.sign(x)
    elif kind == "outliers":
        x[torch.rand(shape, generator=g) < 0.01] *= 1000
    elif kind == "rows":  # per-row magnitudes over 2^-40 .. 2^40
        x *= torch.exp2(torch.randint(-40, 41, (shape[0], 1), generator=g).to(torch.float64))
    elif kind == "wide":  # every element its own binade: saturation and subnormal elements
        x *= torch.exp2(torch.randint(-30, 31, shape, generator=g).to(torch.float64))
    elif kind == "ties":  # short integers: many exact element midpoints and accumulator ties
        x = torch.randint(-64, 65, shape, generator=g).to(torch.float64) / 8
    elif kind == "sparse":  # whole zero blocks and isolated values
        x[torch.rand(shape, generator=g) < 0.9] = 0
    return x.to(torch.float32)


def run_cpp(fmt_id: int, a: torch.Tensor, b: torch.Tensor):
    m, k = a.shape
    n = b.shape[0]
    head = np.array([fmt_id, m, n, k], dtype=np.int32).tobytes()
    data = head + a.numpy().tobytes() + b.numpy().tobytes()
    out = subprocess.run([str(BINARY)], input=data, capture_output=True, check=True, timeout=600)
    raw = out.stdout
    da = np.frombuffer(raw, np.float64, m * k, 0).reshape(m, k)
    db = np.frombuffer(raw, np.float64, n * k, 8 * m * k).reshape(n, k)
    c = np.frombuffer(raw, np.float32, m * n, 8 * (m + n) * k).reshape(m, n)
    return torch.from_numpy(da.copy()), torch.from_numpy(db.copy()), torch.from_numpy(c.copy())


def same_bits(x: torch.Tensor, y: torch.Tensor) -> int:
    """Number of mismatching elements, comparing raw bit patterns."""
    view = torch.int64 if x.dtype == torch.float64 else torch.int32
    return int((x.contiguous().view(view) != y.contiguous().view(view)).sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=4, help="random cases per format and kind")
    parser.add_argument("--kernel", action="store_true", help="also check the CUDA kernel")
    args = parser.parse_args()
    build()
    g = torch.Generator().manual_seed(2026)
    kinds = ["gaussian", "laplace", "outliers", "rows", "wide", "ties", "sparse"]
    failures = 0
    for fmt_id, name in enumerate(FORMAT_IDS):
        fmt = mx.FORMATS[name]
        counts = {"elements": 0, "outputs": 0}
        for kind in kinds:
            for _ in range(args.cases):
                m, n = (int(v) for v in torch.randint(1, 48, (2,), generator=g))
                k = 32 * int(
                    torch.randint(1, 9, (1,), generator=g)
                )  # GPT widths are multiples of 32
                a, b = sample(kind, (m, k), g), sample(kind, (n, k), g)
                da, db, c = run_cpp(fmt_id, a, b)
                qa, qb = mx.quantize(a, fmt), mx.quantize(b, fmt)
                # Elements compare by value: the integer grid has no -0, and a zero element adds
                # nothing to the exact sum whatever its sign.
                bad = int((mx.dequantize(qa) != da).sum() + (mx.dequantize(qb) != db).sum())
                if fmt.element is mx.E5M2:  # quantizer only; see mx.gemm_reference
                    bad_c = 0
                    counts["outputs"] -= m * n
                else:
                    bad_c = same_bits(mx.gemm_reference(qa, qb), c)
                if args.kernel and fmt.element is not mx.E5M2:
                    import kernel

                    # The fused quantizer must give the same grid integers and, on every block
                    # with a nonzero element, the same factors; transposed inputs exercise strides.
                    for x, ref in ((a, qa), (b, qb)):
                        p = kernel.pack(x.T.contiguous().cuda().T, fmt)
                        bad += int((kernel.unpack(p).cpu() != ref.q).sum())
                        live = ref.q.reshape(*ref.scale.shape, -1).ne(0).any(-1)
                        expect = ref.scale.to(torch.float64) * 2.0**fmt.element.grid
                        bad += int((p.factor.cpu().to(torch.float64) != expect)[live].sum())
                    bad_c += same_bits(kernel.gemm(a.cuda(), b.cuda(), fmt).cpu(), c)
                if bad or bad_c:
                    failures += 1
                    print(f"FAIL {name} {kind} {m}x{n}x{k}: {bad} elements, {bad_c} outputs")
                counts["elements"] += (m + n) * k
                counts["outputs"] += m * n
        print(f"{name:12s} {counts['elements']:9d} elements {counts['outputs']:8d} outputs checked")
    print("PASS" if failures == 0 else f"{failures} failing cases")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
