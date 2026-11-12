"""Bit-exact microscaled quantizers and the reference block GEMM.

Every function here mirrors a C++ reference of the numerics study (tools/numerics/datapaths.h,
tb/mx_model.h): the quantizer mirrors to_small/mx_quantize, the GEMM mirrors chained
mx_dot32-style block dot products. xcheck.py proves the match bit for bit.

Representation. A quantized tensor stores each element as a signed integer on its format's
fixed-point grid (2^grid is the smallest subnormal), so an element's value is

    q * 2^grid * block_scale

with one block_scale per `block` consecutive elements along the last (reduction) axis. For OCP MX
formats block_scale = 2^shared (an E8M0 byte); for NVFP4 it is an E4M3 value, and a per-tensor
FP32 decode factor is applied once to the GEMM output.
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Element:
    name: str
    exp_bits: int
    man_bits: int
    max_value: float  # largest finite magnitude; quantization saturates here

    @property
    def bias(self) -> int:
        return (1 << (self.exp_bits - 1)) - 1

    @property
    def grid(self) -> int:
        """Exponent of the smallest subnormal, the LSB of the fixed-point grid."""
        return 1 - self.bias - self.man_bits

    @property
    def emax(self) -> int:
        """Exponent of the largest normal binade (OCP MX v1.0 Table 1)."""
        return math.floor(math.log2(self.max_value))


E4M3 = Element("e4m3", 4, 3, 448.0)
E5M2 = Element("e5m2", 5, 2, 57344.0)
E3M2 = Element("e3m2", 3, 2, 28.0)
E2M3 = Element("e2m3", 2, 3, 7.5)
E2M1 = Element("e2m1", 2, 1, 6.0)


@dataclass(frozen=True)
class Format:
    name: str
    element: Element
    block: int
    nvfp4: bool = False  # E4M3 block scales plus an FP32 tensor scale instead of E8M0


MXFP8 = Format("mxfp8", E4M3, 32)
MXFP8_E5M2 = Format("mxfp8_e5m2", E5M2, 32)
MXFP6_E3M2 = Format("mxfp6_e3m2", E3M2, 32)
MXFP6_E2M3 = Format("mxfp6_e2m3", E2M3, 32)
MXFP4 = Format("mxfp4", E2M1, 32)
NVFP4 = Format("nvfp4", E2M1, 16, nvfp4=True)
FORMATS = {f.name: f for f in (MXFP8, MXFP8_E5M2, MXFP6_E3M2, MXFP6_E2M3, MXFP4, NVFP4)}

NVFP4_RANGE = 6.0 * 448.0  # E2M1 max times E4M3 max (arXiv 2509.25149, Appendix B)


def floor_log2(m: torch.Tensor) -> torch.Tensor:
    """floor(log2 m) for positive m, exactly (frexp gives m = f * 2^e, f in [0.5, 1))."""
    return torch.frexp(m)[1] - 1


def to_grid(y: torch.Tensor, e: Element) -> torch.Tensor:
    """to_small: round y to the nearest-even element, saturating; returns grid integers (int32, int64 for E5M2).

    Every step is exact in y's dtype (scaling by powers of two, rounding to an integer), so
    float32 and float64 give the same result for float32-derived inputs.
    """
    m = y.abs().clamp(max=e.max_value)
    exponent = floor_log2(torch.where(m > 0, m, 1.0)).clamp(min=1 - e.bias)
    step = torch.ldexp(torch.ones_like(m), exponent - e.man_bits)
    q = (torch.round(m / step) * step).clamp(max=e.max_value)  # torch.round is half-to-even
    wide = max_grid_bits(e) > 30  # E5M2 grid integers need 33 bits
    grid = torch.ldexp(q, torch.full_like(exponent, -e.grid)).to(
        torch.int64 if wide else torch.int32
    )
    return torch.where(y < 0, -grid, grid)


@dataclass
class Quantized:
    q: torch.Tensor  # int32 [..., K] grid integers
    scale: torch.Tensor  # [..., K / block] block scale values (exact)
    tensor_decode: float  # NVFP4 per-tensor decode factor, 1.0 for MX
    fmt: Format


def quantize(x: torch.Tensor, fmt: Format) -> Quantized:
    """Quantize along the last axis in blocks of fmt.block (round to nearest even).

    Everything runs in float32. MX quantization only scales by powers of two, so it is exact.
    NVFP4 follows arXiv 2509.25149 Appendix B in float32 arithmetic, operation for operation as
    the C++ reference (nvfp4_quantize): s_enc = 2688 / amax_tensor, s_dec = 1 / s_enc, block
    scale = E4M3(amax_block / 6 * s_enc), element = E2M1(x * (1 / (block scale * s_dec))), each
    step one float32 rounding. The tensor decode factor is s_dec.
    """
    e = fmt.element
    y = x.to(torch.float32).reshape(*x.shape[:-1], -1, fmt.block)
    amax = y.abs().amax(-1, keepdim=True)
    if fmt.nvfp4:
        tensor_amax = float(x.abs().max()) or 1.0
        encode = torch.tensor(NVFP4_RANGE, dtype=torch.float32) / tensor_amax
        code = to_grid(amax / 6.0 * encode, E4M3)
        scale = code.to(torch.float32) * 2.0**E4M3.grid
        s_dec = 1.0 / encode
        inverse = 1.0 / (torch.where(scale > 0, scale, 1.0) * s_dec)
        q = torch.where(scale > 0, to_grid(y * inverse, e), 0)
        decode = float(s_dec)
    else:
        # OCP MX v1.0 section 6.3: shared exponent floor(log2 amax) - emax_elem, as E8M0.
        shared = torch.where(amax > 0, floor_log2(torch.where(amax > 0, amax, 1.0)) - e.emax, -127)
        shared = shared.clamp(-127, 127)
        scale = torch.ldexp(torch.ones_like(amax), shared)
        q = to_grid(torch.ldexp(y, -shared), e)
        decode = 1.0
    return Quantized(q.reshape(x.shape), scale.squeeze(-1), decode, fmt)


def dequantize(t: Quantized) -> torch.Tensor:
    """Exact float64 values of the quantized elements (without the NVFP4 tensor factor)."""
    b = t.fmt.block
    v = t.q.reshape(*t.q.shape[:-1], -1, b).to(torch.float64) * 2.0**t.fmt.element.grid
    return (v * t.scale.unsqueeze(-1)).reshape(t.q.shape)


def max_grid_bits(e: Element) -> int:
    """Bits of the largest grid integer (448 * 2^9 < 2^18 for E4M3)."""
    return int(e.max_value * 2.0**-e.grid).bit_length()


def round_to_odd_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a + b in float64 rounded to odd: if inexact, the neighbour with an odd last bit.

    Rounding to odd at 53 bits and then to nearest at 24 bits equals one rounding to nearest at
    24 bits (Boldo and Melquiond, IEEE Trans. Computers 57(4), 2008), so this gives the correctly
    rounded FP32 sum of two float64 values.
    """
    s = a + b
    bb = s - a
    err = (a - (s - bb)) + (b - bb)  # TwoSum: s + err == a + b exactly
    bits = s.view(torch.int64)
    even = (bits & 1) == 0
    toward_larger = (err > 0) == (s > 0)  # moving |s| up or down one unit in the last place
    bits = torch.where((err != 0) & even, bits + torch.where(toward_larger, 1, -1), bits)
    return bits.view(torch.float64)


def gemm_reference(a: Quantized, b: Quantized) -> torch.Tensor:
    """C = A B^T as the hardware computes it, on the CPU.

    For each 32 elements of K: the exact integer dot product of the grid values, times both block
    scales, plus the FP32 accumulator, rounded once to FP32, chained from +0 in K order. The
    NVFP4 unit (nvfp4_dot32) sums its two 16-element blocks, each times its own scale product,
    exactly before that one rounding; the host then multiplies by s_dec_a * s_dec_b in FP32.
    """
    fmt = a.fmt
    m, k = a.q.shape
    n = b.q.shape[0]
    blocks = k // fmt.block
    # The block sum must be exact in int64 and then in float64. E5M2 spans 32 binades, so its
    # 32-term sum needs up to 69 bits; it is not emulated (E4M3 needs 41, FP4 and FP6 at most 23).
    if 2 * (max_grid_bits(fmt.element) + 1) + fmt.block.bit_length() > 53:
        raise ValueError(f"{fmt.name} block sums do not fit 53 bits")
    # Grid integers in float64: every product and partial block sum is an integer below 2^53,
    # so a float64 matrix product (CPU or GPU, any summation order) is exact.
    qa = a.q.to(torch.float64).reshape(m, blocks, fmt.block)
    qb = b.q.to(torch.float64).reshape(n, blocks, fmt.block)
    grid = 2.0 ** (2 * fmt.element.grid)
    sa, sb = a.scale.to(torch.float64), b.scale.to(torch.float64)
    acc = torch.zeros(m, n, dtype=torch.float32, device=a.q.device)
    per_unit = 32 // fmt.block
    for i in range(0, blocks, per_unit):
        # Each block term is exact (two short significands times the sum). The NVFP4 pair spans
        # under 48 bits (block sums < 2^12, scales 2^-9..448), so its float64 sum is exact too.
        term = sum(
            (qa[:, j] @ qb[:, j].T) * (sa[:, j, None] * sb[None, :, j] * grid)
            for j in range(i, i + per_unit)
        )
        acc = round_to_odd_add(acc.to(torch.float64), term).to(torch.float32)
    if fmt.nvfp4:
        f32 = torch.float32
        decode = torch.tensor(a.tensor_decode, dtype=f32) * torch.tensor(b.tensor_decode, dtype=f32)
        acc = acc * decode.to(acc.device)
    return acc
