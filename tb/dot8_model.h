// Reference for the V6 fused dot product (docs/training-v6.md) and an exact-sum oracle.
#pragma once
#include <array>
#include <cstdint>

struct Dot8Result { uint32_t bits; unsigned flags; bool truncated; };

namespace dot8 {

constexpr uint32_t canonical_nan = 0x7fc00000u;

struct Operand { bool sign, zero, inf, nan, snan; uint32_t sig; int lsb; };

// Width is the stored fraction width; bias - fraction width gives the LSB exponent.
inline Operand decode(uint32_t raw, unsigned fraction, unsigned exponent_bits, int bias) {
    unsigned field = (raw >> fraction) & ((1u << exponent_bits) - 1);
    uint32_t frac = raw & ((1u << fraction) - 1);
    bool top = field == (1u << exponent_bits) - 1;
    Operand x{bool(raw >> (fraction + exponent_bits)), field == 0 && frac == 0, top && frac == 0,
              top && frac != 0, top && frac != 0 && !(frac >> (fraction - 1))};
    x.sig = (field != 0 ? 1u << fraction : 0) | frac;
    x.lsb = int(field ? field : 1) - bias - int(fraction);
    return x;
}

struct Term { bool sign; uint32_t sig; int lsb; unsigned width; };

struct Terms {
    std::array<Term, 9> term;
    bool nan = false, invalid = false, pos_inf = false, neg_inf = false, all_zero = true, all_negative = true;
};

inline Terms terms(const std::array<uint16_t, 8> &a, const std::array<uint16_t, 8> &b, uint32_t c) {
    Terms t;
    for (unsigned k = 0; k < 8; ++k) {
        Operand x = decode(a[k], 7, 8, 127), y = decode(b[k], 7, 8, 127);
        bool sign = x.sign != y.sign;
        t.nan |= x.nan || y.nan;
        t.invalid |= x.snan || y.snan || (x.inf && y.zero) || (y.inf && x.zero);
        if ((x.inf || y.inf) && !x.nan && !y.nan && !x.zero && !y.zero) (sign ? t.neg_inf : t.pos_inf) = true;
        t.term[k] = {sign, x.sig * y.sig, x.lsb + y.lsb, 16};
        t.all_zero &= t.term[k].sig == 0;
        t.all_negative &= sign;
    }
    Operand z = decode(c, 23, 8, 127);
    t.nan |= z.nan;
    t.invalid |= z.snan;
    if (z.inf) (z.sign ? t.neg_inf : t.pos_inf) = true;
    t.term[8] = {z.sign, z.sig, z.lsb, 24};
    t.all_zero &= z.sig == 0;
    t.all_negative &= z.sign;
    t.invalid |= t.pos_inf && t.neg_inf;
    return t;
}

// Rounds sign * mag * 2^lsb to FP32 nearest-even; mag < 2^63.
inline uint32_t round_fp32(bool sign, uint64_t mag, int lsb, bool &inexact, bool &tiny) {
    int lead = 63 - __builtin_clzll(mag);
    int exponent = lead + lsb;
    tiny = exponent < -126;
    int g = exponent - 23 > -149 ? exponent - 23 : -149;
    int shift = g - lsb;
    uint64_t sig;
    inexact = false;
    if (shift <= 0) sig = mag << -shift;
    else {
        uint64_t kept = shift >= 64 ? 0 : mag >> shift;
        bool round = shift <= 64 && ((mag >> (shift - 1)) & 1);
        bool sticky = shift >= 65 ? mag != 0 : (shift >= 2 && (mag & ((uint64_t(1) << (shift - 1)) - 1)) != 0);
        inexact = round || sticky;
        sig = kept + (round && (sticky || (kept & 1)));
    }
    if (sig >> 24) { sig >>= 1; ++g; }
    unsigned field = sig >> 23 ? unsigned(g + 150) : 0;
    if (g + 150 >= 255) { inexact = true; return uint32_t(sign) << 31 | 0x7f800000u; }
    return uint32_t(sign) << 31 | field << 23 | (uint32_t(sig) & 0x7fffffu);
}

}  // namespace dot8

// The specification: 32-bit frame below the largest term's top, truncation, one rounding.
inline Dot8Result dot8_spec(const std::array<uint16_t, 8> &a, const std::array<uint16_t, 8> &b, uint32_t c) {
    using namespace dot8;
    Terms t = terms(a, b, c);
    if (t.nan || t.invalid) return {canonical_nan, t.invalid ? 16u : 0u, false};
    if (t.pos_inf || t.neg_inf) return {t.neg_inf ? 0xff800000u : 0x7f800000u, 0, false};
    if (t.all_zero) return {t.all_negative ? 0x80000000u : 0u, 0, false};
    int top = -100000;
    for (const Term &x : t.term)
        if (x.sig && x.lsb + int(x.width) > top) top = x.lsb + int(x.width);
    int64_t sum = 0;
    bool truncated = false;
    for (const Term &x : t.term) {
        if (!x.sig) continue;
        uint64_t justified = uint64_t(x.sig) << (32 - x.width);
        int distance = top - (x.lsb + int(x.width));
        uint64_t aligned = distance >= 32 ? 0 : justified >> distance;
        truncated |= distance >= 32 ? justified != 0 : (aligned << distance) != justified;
        sum += x.sign ? -int64_t(aligned) : int64_t(aligned);
    }
    if (sum == 0) return {0u, truncated ? 1u : 0u, truncated};
    bool inexact, tiny;
    uint64_t mag = sum < 0 ? uint64_t(-sum) : uint64_t(sum);
    uint32_t bits = round_fp32(sum < 0, mag, top - 32, inexact, tiny);
    inexact |= truncated;
    bool overflow = (bits & 0x7fffffffu) == 0x7f800000u;
    return {bits, (overflow ? 4u : 0u) | (tiny && inexact ? 2u : 0u) | (inexact ? 1u : 0u), truncated};
}

// Independent oracle: exact two's-complement sum in 640 bits, then correctly rounded.
inline Dot8Result dot8_exact(const std::array<uint16_t, 8> &a, const std::array<uint16_t, 8> &b, uint32_t c) {
    using namespace dot8;
    Terms t = terms(a, b, c);
    if (t.nan || t.invalid) return {canonical_nan, t.invalid ? 16u : 0u, false};
    if (t.pos_inf || t.neg_inf) return {t.neg_inf ? 0xff800000u : 0x7f800000u, 0, false};
    constexpr int words = 10, offset = 300;  // bit i has weight 2^(i - offset)
    std::array<uint64_t, words> acc{};
    for (const Term &x : t.term) {
        if (!x.sig) continue;
        std::array<uint64_t, words> v{};
        int pos = x.lsb + offset;
        v[pos / 64] = uint64_t(x.sig) << (pos % 64);
        if (pos % 64) v[pos / 64 + 1] = uint64_t(x.sig) >> (64 - pos % 64);
        if (x.sign) {
            unsigned carry = 1;
            for (auto &w : v) { w = ~w + carry; carry = carry && w == 0; }
        }
        unsigned carry = 0;
        for (int i = 0; i < words; ++i) {
            uint64_t s = acc[i] + v[i];
            unsigned next = s < acc[i];
            acc[i] = s + carry;
            carry = next | (acc[i] < s);
        }
    }
    bool negative = acc[words - 1] >> 63;
    if (negative) {
        unsigned carry = 1;
        for (auto &w : acc) { w = ~w + carry; carry = carry && w == 0; }
    }
    int lead = -1;
    for (int i = words - 1; i >= 0 && lead < 0; --i)
        if (acc[i]) lead = 64 * i + 63 - __builtin_clzll(acc[i]);
    if (lead < 0) return {t.all_zero && t.all_negative ? 0x80000000u : 0u, 0, false};
    // Keep the top 40 bits plus a sticky bit; the remainder only affects stickiness.
    int low = lead - 39 > 0 ? lead - 39 : 0;
    uint64_t mag = 0;
    for (int bit = lead; bit >= low; --bit) mag = mag << 1 | ((acc[bit / 64] >> (bit % 64)) & 1);
    bool rest = false;
    for (int bit = 0; bit < low; ++bit) rest |= (acc[bit / 64] >> (bit % 64)) & 1;
    mag = mag << 1 | rest;
    bool inexact, tiny;
    uint32_t bits = dot8::round_fp32(negative, mag, low - offset - 1, inexact, tiny);
    bool overflow = (bits & 0x7fffffffu) == 0x7f800000u;
    return {bits, (overflow ? 4u : 0u) | (tiny && inexact ? 2u : 0u) | (inexact ? 1u : 0u), false};
}
