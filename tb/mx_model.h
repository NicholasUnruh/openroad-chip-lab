// Reference for the OCP MX block dot product (docs/numerics-study.md): correctly rounded
// c + 2^(sa + sb - 254) * sum of 32 element products, from the exact-sum oracle.
// This header is the bit-exact ground truth for the MX formats; the training emulation and the RTL
// (rtl/mx_dot32_fp32.v, ELEM = the enum value) must match it.
#pragma once
#include "dot8_model.h"
#include <array>
#include <cstdint>

// OCP MX element types. The values are stable: they equal the RTL ELEM parameter.
// | Type | Bits | Bias | Max   | Specials                                         |
// | e4m3 | 8    | 7    | 448   | NaN at S.1111.111, no infinities                 |
// | e2m1 | 4    | 1    | 6     | none                                             |
// | e5m2 | 8    | 15   | 57344 | IEEE inf and NaN; signaling if mantissa MSB is 0 |
// | e3m2 | 6    | 3    | 28    | none                                             |
// | e2m3 | 6    | 1    | 7.5   | none                                             |
enum class MxElement { e4m3 = 0, e2m1 = 1, e5m2 = 2, e3m2 = 3, e2m3 = 4 };

struct MxFormat {
    unsigned exponent, mantissa;  // field widths
    double max_value;             // largest finite magnitude
    int emax;                     // exponent of max_value; shared scale = floor(log2 amax) - emax
};

inline MxFormat mx_format(MxElement e) {
    switch (e) {
        case MxElement::e4m3:
            return {4, 3, 448.0, 8};
        case MxElement::e2m1:
            return {2, 1, 6.0, 2};
        case MxElement::e5m2:
            return {5, 2, 57344.0, 15};
        case MxElement::e3m2:
            return {3, 2, 28.0, 4};
        default:
            return {2, 3, 7.5, 2};
    }
}

inline dot8::Operand mx_decode(uint8_t raw, MxElement e) {
    if (e == MxElement::e4m3) return dot8::decode_input(raw, dot8_e4m3);
    if (e == MxElement::e5m2) return dot8::decode_input(raw, dot8_e5m2);
    MxFormat f = mx_format(e);
    dot8::Operand x = dot8::decode(raw, f.mantissa, f.exponent, (1 << (f.exponent - 1)) - 1);
    x.inf = x.nan = x.snan = false;
    return x;
}

// Specials: a scale of 0xFF or a NaN element gives the canonical NaN; signaling NaNs (E5M2, c),
// infinity times zero and opposite infinities also raise invalid (16). Otherwise an infinite
// product or c gives that infinity, exactly.
inline Dot8Result mx_dot32(const std::array<uint8_t, 32> &a, const std::array<uint8_t, 32> &b,
                           uint8_t scale_a, uint8_t scale_b, uint32_t c, MxElement e) {
    using namespace dot8;
    std::array<Term, 33> term;
    bool nan = scale_a == 0xff || scale_b == 0xff, invalid = false, pos_inf = false,
         neg_inf = false, all_zero = true, all_negative = true;
    int scale = int(scale_a) + int(scale_b) - 254;
    for (unsigned k = 0; k < 32; ++k) {
        Operand x = mx_decode(a[k], e), y = mx_decode(b[k], e);
        nan |= x.nan || y.nan;
        invalid |= x.snan || y.snan || (x.inf && y.zero) || (y.inf && x.zero);
        bool sign = x.sign != y.sign;
        if ((x.inf || y.inf) && !x.nan && !y.nan && !x.zero && !y.zero)
            (sign ? neg_inf : pos_inf) = true;
        term[k] = {sign, x.sig * y.sig, x.lsb + y.lsb + scale, 0};
        all_zero &= term[k].sig == 0;
        all_negative &= sign;
    }
    Operand z = decode(c, 23, 8, 127);
    if (z.inf) (z.sign ? neg_inf : pos_inf) = true;
    invalid |= z.snan || (pos_inf && neg_inf);
    if (nan || z.nan || invalid) return {canonical_nan, invalid ? 16u : 0u, false};
    if (pos_inf || neg_inf) return {neg_inf ? 0xff800000u : 0x7f800000u, 0, false};
    term[32] = {z.sign, z.sig, z.lsb, 0};
    return exact_round(term.data(), term.data() + term.size(), all_zero && z.sig == 0,
                       all_negative && z.sign);
}

// NVFP4 ("Pretraining Large Language Models with NVFP4", NVIDIA, arXiv:2509.25149, Section 2 and
// Appendix B): E2M1 elements, one E4M3 decode scale per 16 elements, and an FP32 per-tensor scale
// that the host applies to the GEMM output. One unit call covers 32 elements, two blocks:
// correctly rounded c + sum over blocks of sa[blk] * sb[blk] * sum of 16 element products, all
// exact before one rounding (the paper's Eq. 5 per block). Scale sign bits are ignored (scales are
// positive); a scale with low seven bits 0x7f is the E4M3 NaN and gives the canonical NaN.
inline Dot8Result nvfp4_dot32(const std::array<uint8_t, 32> &a, const std::array<uint8_t, 32> &b,
                              const std::array<uint8_t, 2> &scale_a,
                              const std::array<uint8_t, 2> &scale_b, uint32_t c) {
    using namespace dot8;
    std::array<Term, 33> term;
    bool nan = false, all_zero = true, all_negative = true;
    std::array<Operand, 2> sa, sb;
    for (unsigned blk = 0; blk < 2; ++blk) {
        sa[blk] = decode_input(scale_a[blk] & 0x7fu, dot8_e4m3);
        sb[blk] = decode_input(scale_b[blk] & 0x7fu, dot8_e4m3);
        nan |= sa[blk].nan || sb[blk].nan;
    }
    for (unsigned k = 0; k < 32; ++k) {
        Operand x = mx_decode(a[k], MxElement::e2m1), y = mx_decode(b[k], MxElement::e2m1);
        const Operand &s = sa[k / 16], &t = sb[k / 16];
        bool sign = x.sign != y.sign;
        term[k] = {sign, x.sig * y.sig * s.sig * t.sig, x.lsb + y.lsb + s.lsb + t.lsb, 0};
        all_zero &= term[k].sig == 0;
        all_negative &= sign;
    }
    Operand z = decode(c, 23, 8, 127);
    if (nan || z.nan) return {canonical_nan, z.snan ? 16u : 0u, false};
    if (z.inf) return {z.sign ? 0xff800000u : 0x7f800000u, 0, false};
    term[32] = {z.sign, z.sig, z.lsb, 0};
    return exact_round(term.data(), term.data() + term.size(), all_zero && z.sig == 0,
                       all_negative && z.sign);
}
