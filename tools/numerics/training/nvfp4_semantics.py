"""How much the NVFP4 unit definition changes GEMM outputs: the earlier provisional definition
(one FP32 rounding per 16-element block, element = x * s_enc / scale, decode amax_a/2688 *
amax_b/2688 in float64) against the hardware unit (mx.py: one rounding per 32 elements, element =
x * (1 / (scale * s_dec)), decode s_dec_a * s_dec_b in FP32).

usage: uv run --extra cpu python nvfp4_semantics.py [--out file]
"""

import argparse

import torch

import mx


def provisional(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Grid integers of a and the GEMM A B^T under the earlier per-16 definition."""

    def quantize(x):
        amax = float(x.abs().max()) or 1.0
        encode = torch.tensor(mx.NVFP4_RANGE, dtype=torch.float32) / amax
        y = x.reshape(x.shape[0], -1, 16)
        code = mx.to_grid(y.abs().amax(-1, keepdim=True) / 6.0 * encode, mx.E4M3)
        scale = code.to(torch.float32) * 2.0**mx.E4M3.grid
        q = torch.where(scale > 0, mx.to_grid(y * encode / torch.where(scale > 0, scale, 1.0),
                                              mx.E2M1), 0)  # fmt: skip
        return q, scale.squeeze(-1).double(), amax / mx.NVFP4_RANGE

    (qa, sa, da), (qb, sb, db) = quantize(a), quantize(b)
    acc = torch.zeros(a.shape[0], b.shape[0], dtype=torch.float32)
    for j in range(qa.shape[1]):
        exact = (qa[:, j].long() @ qb[:, j].long().T).double()
        term = exact * (sa[:, j, None] * sb[None, :, j] * 2.0 ** (2 * mx.E2M1.grid))
        acc = mx.round_to_odd_add(acc.double(), term).float()
    return qa.reshape(a.shape), acc * torch.tensor(da * db, dtype=torch.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out")
    args = p.parse_args()
    g = torch.Generator().manual_seed(1)
    elements = differing_q = outputs = differing_c = 0
    worst = 0.0
    for _ in range(20):
        a = torch.randn(64, 768, generator=g) * (1 + 10 * torch.rand(64, 1, generator=g))
        b = torch.randn(64, 768, generator=g)
        qa, qb = mx.quantize(a, mx.NVFP4), mx.quantize(b, mx.NVFP4)
        old_q, old_c = provisional(a, b)
        c = mx.gemm_reference(qa, qb)
        elements += qa.q.numel()
        differing_q += int((qa.q != old_q).sum())
        outputs += c.numel()
        differing_c += int((c != old_c).sum())
        worst = max(worst, float(((c - old_c).abs() / c.abs().clamp(min=1e-30)).max()))
    text = (
        f"elements (A operands) differing: {differing_q} of {elements}\n"
        f"GEMM outputs differing:          {differing_c} of {outputs}"
        f" ({100 * differing_c / outputs:.1f}%)\n"
        f"largest relative difference:     {worst:.2e}\n"
    )
    print(text, end="")
    if args.out:
        open(args.out, "w").write(text)


if __name__ == "__main__":
    main()
