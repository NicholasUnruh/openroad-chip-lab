"""Bit-exact block GEMM on the GPU: exact integer block dot products on int8 tensor cores.

Element values are integers on the format's fixed-point grid (mx.py), so a block dot product is
an integer dot product, the same aligned integer sum the hardware unit forms. Grid integers wider
than int8 are split into 6-bit pieces, q = sum_i piece_i * 64^i, with a signed top piece:

    E2M1 (|q| <= 12), E2M3 (60): one piece    E3M2 (448): two    E4M3 (229376): three

The int8 x int8 -> int32 tensor-core products of all piece pairs are exact; pairs with the same
weight 64^(i+j) are summed in int32. Per K block the kernel then adds the exact sum times both
block scales to the FP32 accumulator with one rounding to nearest even:

- one or two pieces: the block sum is below 2^24, so it is exact in FP32, scaling by the block
  scales is exact, and a single FP32 add rounds once;
- three pieces (E4M3, up to 2^41): integer carries split the sum exactly into hi * 2^18 + lo,
  both exact in FP32, and acc + hi + lo is rounded once with the Boldo-Melquiond three-term algorithm (two TwoSums, an
  addition rounded to odd, a final rounded addition).
"""

import os
from dataclasses import dataclass

import torch

# No multiply-add contraction: every product here is exact anyway, but this keeps the generated
# arithmetic exactly the one written.
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "0")
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import mx  # noqa: E402

PIECE_BITS = 6


def piece_count(e: mx.Element) -> int:
    bits = mx.max_grid_bits(e)
    return 1 if bits <= 7 else 1 + -(-(bits - 7) // PIECE_BITS)


@triton.jit
def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


@triton.jit
def _round_to_odd_add(a, b):
    s, err = _two_sum(a, b)
    bits = s.to(tl.int32, bitcast=True)
    step = tl.where((err > 0) == (s > 0), 1, -1)
    bits = tl.where((err != 0) & ((bits & 1) == 0), bits + step, bits)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _block_sum(
    a_ptrs,
    b_ptrs,
    a_piece,
    b_piece,
    mask_a,
    mask_b,
    half,
    PIECES: tl.constexpr,
    HALVES: tl.constexpr,
):
    """Exact dot product of one 32-wide K slice (or one 16-wide half of it) as int32 (hi, lo).

    One or two pieces: the sum itself (lo unused). Three pieces: S = hi * 2^18 + lo.
    """
    rk = tl.arange(0, 32)
    if PIECES == 1:
        a0 = tl.load(a_ptrs, mask=mask_a, other=0)
        if HALVES:
            a0 = tl.where((rk // 16 == half)[None, :], a0, 0)
        s = tl.dot(a0, tl.load(b_ptrs, mask=mask_b, other=0), out_dtype=tl.int32)
        return s, s
    else:
        a0 = tl.load(a_ptrs, mask=mask_a, other=0)
        a1 = tl.load(a_ptrs + a_piece, mask=mask_a, other=0)
        b0 = tl.load(b_ptrs, mask=mask_b, other=0)
        b1 = tl.load(b_ptrs + b_piece, mask=mask_b, other=0)
        c0 = tl.dot(a0, b0, out_dtype=tl.int32)
        c1 = tl.dot(a0, b1, out_dtype=tl.int32) + tl.dot(a1, b0, out_dtype=tl.int32)
        c2 = tl.dot(a1, b1, out_dtype=tl.int32)
        if PIECES == 2:
            s = c0 + (c1 << 6) + (c2 << 12)
            return s, s
        else:
            a2 = tl.load(a_ptrs + 2 * a_piece, mask=mask_a, other=0)
            b2 = tl.load(b_ptrs + 2 * b_piece, mask=mask_b, other=0)
            c2 += tl.dot(a0, b2, out_dtype=tl.int32) + tl.dot(a2, b0, out_dtype=tl.int32)
            c3 = tl.dot(a1, b2, out_dtype=tl.int32) + tl.dot(a2, b1, out_dtype=tl.int32)
            c4 = tl.dot(a2, b2, out_dtype=tl.int32)
            # Normalize to S = hi * 2^18 + lo with 0 <= lo < 2^18 and |hi| < 2^23, so both are
            # exact in FP32, using int32 only (|c0 + c1 2^6 + c2 2^12| < 1.47e9 < 2^31).
            lo = c0 + (c1 << 6) + (c2 << 12)
            return c3 + (c4 << 6) + (lo >> 18), lo & 0x3FFFF


@triton.jit
def _accumulate(acc, hi, lo, fa, fb, PIECES: tl.constexpr):
    """acc + S * fa * fb rounded once to FP32; the products by the scales are exact."""
    if PIECES < 3:
        return acc + hi.to(tl.float32) * fa[:, None] * fb[None, :]
    else:
        b = hi.to(tl.float32) * 262144.0 * fa[:, None] * fb[None, :]
        c = lo.to(tl.float32) * fa[:, None] * fb[None, :]
        uh, ul = _two_sum(b, c)
        th, tl_ = _two_sum(acc, uh)
        return th + _round_to_odd_add(tl_, ul)


@triton.autotune(
    configs=[
        triton.Config({"BM": bm, "BN": bn}, num_warps=w, num_stages=s)
        for bm, bn in ((64, 64), (128, 64), (64, 128), (128, 128))
        for w in (4, 8)
        for s in (2, 3)
    ],
    key=["BLOCK", "PIECES"],  # tuned once per format, on the first (training-sized) call
)
@triton.jit
def _gemm_kernel(
    a_ptr,
    b_ptr,
    fa_ptr,
    fb_ptr,
    c_ptr,
    M,
    N,
    K,
    a_piece,
    b_piece,
    BLOCK: tl.constexpr,
    PIECES: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, 32)
    mask_a = (rm < M)[:, None]
    mask_b = (rn < N)[None, :]
    blocks = K // BLOCK
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, 32):
        a_ptrs = a_ptr + rm[:, None].to(tl.int64) * K + (k0 + rk)[None, :]
        b_ptrs = b_ptr + rn[None, :].to(tl.int64) * K + (k0 + rk)[:, None]
        if BLOCK == 32:
            hi, lo = _block_sum(a_ptrs, b_ptrs, a_piece, b_piece, mask_a, mask_b, 0, PIECES, False)
            fa = tl.load(fa_ptr + rm * blocks + k0 // 32, mask=rm < M, other=0.0)
            fb = tl.load(fb_ptr + rn * blocks + k0 // 32, mask=rn < N, other=0.0)
            acc = _accumulate(acc, hi, lo, fa, fb, PIECES)
        else:  # two 16-element blocks per slice, in K order
            for half in tl.static_range(2):
                hi, lo = _block_sum(
                    a_ptrs, b_ptrs, a_piece, b_piece, mask_a, mask_b, half, PIECES, True
                )
                fa = tl.load(fa_ptr + rm * blocks + k0 // 16 + half, mask=rm < M, other=0.0)
                fb = tl.load(fb_ptr + rn * blocks + k0 // 16 + half, mask=rn < N, other=0.0)
                acc = _accumulate(acc, hi, lo, fa, fb, PIECES)
    c_ptrs = c_ptr + rm[:, None].to(tl.int64) * N + rn[None, :]
    tl.store(c_ptrs, acc, mask=mask_a & mask_b)


# ---------------------------------------------------------------------------------------------
# Fused quantizer: the same arithmetic as mx.quantize, one pass, writing int8 pieces directly.


@triton.jit
def _pow2_f32(e):
    """2^e as FP32, exact for integer e in [-149, 127] (subnormals from their bit pattern)."""
    bits = tl.where(e >= -126, (e + 127) << 23, 1 << tl.minimum(tl.maximum(e + 149, 0), 30))
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _floor_log2_f32(m):
    """floor(log2 m) for m > 0, from the FP32 exponent field (subnormals rescaled by 2^64)."""
    sub = m < 1.1754943508222875e-38
    m = tl.where(sub, m * 18446744073709551616.0, m)
    field = (m.to(tl.int32, bitcast=True) >> 23) & 0xFF
    return field - 127 - tl.where(sub, 64, 0)


@triton.jit
def _to_grid(y, EXP_BITS: tl.constexpr, MAN_BITS: tl.constexpr, MAX_VALUE: tl.constexpr):
    """to_small on FP32 y: nearest-even element, saturating, as a grid integer."""
    BIAS: tl.constexpr = (1 << (EXP_BITS - 1)) - 1
    GRID: tl.constexpr = 1 - BIAS - MAN_BITS
    m = tl.minimum(tl.abs(y), MAX_VALUE)
    e = tl.maximum(_floor_log2_f32(tl.where(m > 0, m, 1.0)), 1 - BIAS)
    r = m * _pow2_f32(MAN_BITS - e)  # exact: below 2^(MAN_BITS + 1)
    r = (r + 12582912.0) - 12582912.0  # 1.5 * 2^23: round half to even
    q = tl.minimum(r * _pow2_f32(e - MAN_BITS), MAX_VALUE)
    g = (q * _pow2_f32(tl.zeros_like(e) - GRID)).to(tl.int32)
    return tl.where(y < 0, -g, g)


@triton.jit
def _quantize_kernel(
    x_ptr,
    stride_r,
    stride_k,
    R,
    K,
    pieces_ptr,
    factor_ptr,
    encode_ptr,
    EXP_BITS: tl.constexpr,
    MAN_BITS: tl.constexpr,
    MAX_VALUE: tl.constexpr,
    EMAX: tl.constexpr,
    BLOCK: tl.constexpr,
    PIECES: tl.constexpr,
    NVFP4: tl.constexpr,
    BR: tl.constexpr,
):
    BIAS: tl.constexpr = (1 << (EXP_BITS - 1)) - 1
    GRID: tl.constexpr = 1 - BIAS - MAN_BITS
    rr = tl.program_id(0) * BR + tl.arange(0, BR)
    rk = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = (rr < R)[:, None]
    x = tl.load(
        x_ptr + rr[:, None].to(tl.int64) * stride_r + rk[None, :].to(tl.int64) * stride_k,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=1)
    if NVFP4:  # float32, rounded division as in mx.quantize (Triton's "/" is approximate)
        encode = tl.load(encode_ptr)
        code = _to_grid(tl.math.div_rn(amax, 6.0) * encode, 4, 3, 448.0)
        scale = code.to(tl.float32) * 0.001953125  # E4M3 grid 2^-9
        safe = tl.where(scale > 0, scale, 1.0)
        y = tl.math.div_rn(x * encode, safe[:, None])
        q = tl.where((scale > 0)[:, None], _to_grid(y, EXP_BITS, MAN_BITS, MAX_VALUE), 0)
        factor = scale * (2.0**GRID)
    else:
        shared = tl.where(amax > 0, _floor_log2_f32(tl.where(amax > 0, amax, 1.0)) - EMAX, -127)
        shared = tl.minimum(tl.maximum(shared, -127), 127)
        q = _to_grid(x * _pow2_f32(-shared)[:, None], EXP_BITS, MAN_BITS, MAX_VALUE)
        # Zero blocks get factor 1 (their sum is 0 anyway) so the range check sees live blocks.
        factor = tl.where(amax > 0, _pow2_f32(shared + GRID), 1.0)
    out = pieces_ptr + rr[:, None].to(tl.int64) * K + rk[None, :]
    for i in tl.static_range(PIECES):
        if i == PIECES - 1:
            piece = q >> (6 * i)
        else:
            piece = (q >> (6 * i)) & 63
        tl.store(out + i * R * K, piece.to(tl.int8), mask=mask)
    tl.store(factor_ptr + rr * (K // BLOCK) + tl.program_id(1), factor, mask=rr < R)


@dataclass
class Packed:
    """A quantized operand in the kernel's layout: int8 pieces [P, R, K], FP32 block factors
    [R, K / block] (scale * 2^grid), and the NVFP4 tensor decode factor as a float64 tensor."""

    pieces: torch.Tensor
    factor: torch.Tensor
    decode: torch.Tensor | None
    fmt: mx.Format


def pack(x: torch.Tensor, fmt: mx.Format) -> Packed:
    """Quantize a 2-D CUDA tensor along its last axis; any strides (transposes are free)."""
    r, k = x.shape
    e = fmt.element
    assert mx.max_grid_bits(e) <= 30, "E5M2 is quantizer-only (see mx.gemm_reference)"
    count = piece_count(e)
    pieces = torch.empty(count, r, k, dtype=torch.int8, device=x.device)
    factor = torch.empty(r, k // fmt.block, dtype=torch.float32, device=x.device)
    decode = None
    encode = factor  # unused placeholder pointer for MX
    if fmt.nvfp4:
        amax = x.abs().max().float()
        amax = torch.where(amax > 0, amax, 1.0)
        encode = torch.tensor(mx.NVFP4_RANGE, dtype=torch.float32, device=x.device) / amax
        decode = amax.to(torch.float64) / mx.NVFP4_RANGE
    br = 64
    grid = (triton.cdiv(r, br), k // fmt.block)
    _quantize_kernel[grid](
        x,
        x.stride(0),
        x.stride(1),
        r,
        k,
        pieces,
        factor,
        encode,
        EXP_BITS=e.exp_bits,
        MAN_BITS=e.man_bits,
        MAX_VALUE=e.max_value,
        EMAX=e.emax,
        BLOCK=fmt.block,
        PIECES=count,
        NVFP4=fmt.nvfp4,
        BR=br,
    )
    return Packed(pieces, factor, decode, fmt)


def unpack(p: Packed) -> torch.Tensor:
    """Grid integers back from the pieces (for checking pack against mx.quantize)."""
    q = p.pieces[-1].to(torch.int32)
    for i in reversed(range(p.pieces.shape[0] - 1)):
        q = q * 64 + p.pieces[i].to(torch.int32)
    return q


# Count of GEMMs whose block scales left the exact FP32 range; the trainer checks it each step.
range_errors = None


def gemm_packed(a: Packed, b: Packed) -> torch.Tensor:
    """C = A B^T for packed A [M, K], B [N, K]; matches mx.gemm_reference bit for bit."""
    global range_errors
    fmt = a.fmt
    _, m, k = a.pieces.shape
    n = b.pieces.shape[1]
    assert k % 32 == 0, "K must be a multiple of 32"
    if not fmt.nvfp4:  # MX: the smallest result LSB, factor_a * factor_b, must be >= 2^-149
        bad = (a.factor.min() * b.factor.min() < 2.0**-149).to(torch.int32)
        range_errors = bad if range_errors is None else range_errors + bad
    c = torch.empty(m, n, dtype=torch.float32, device=a.pieces.device)
    grid = lambda meta: (triton.cdiv(m, meta["BM"]), triton.cdiv(n, meta["BN"]))  # noqa: E731
    _gemm_kernel[grid](
        a.pieces,
        b.pieces,
        a.factor,
        b.factor,
        c,
        m,
        n,
        k,
        m * k,
        n * k,
        BLOCK=fmt.block,
        PIECES=a.pieces.shape[0],
    )
    if fmt.nvfp4:
        c *= (a.decode * b.decode).to(torch.float32)
    return c


def gemm(a: torch.Tensor, b: torch.Tensor, fmt: mx.Format) -> torch.Tensor:
    return gemm_packed(pack(a, fmt), pack(b, fmt))
