// C++ side of the randomized cross-check (xcheck.py): quantizes and multiplies FP32 matrices with
// the numerics study's references and writes the results for a bit-for-bit comparison.
//
// stdin:  int32 format, M, N, K; float A[M*K]; float B[N*K]   (C = A B^T, blocks along K)
// stdout: double dequantized A[M*K], B[N*K]; float C[M*N]
// Formats: 0 MXFP8 E4M3, 1 MXFP8 E5M2, 2 MXFP6 E3M2, 3 MXFP6 E2M3, 4 MXFP4 E2M1, 5 NVFP4.
//
// MXFP8 E4M3 and MXFP4 GEMMs come from mx_chain (quantizer plus mx_dot32, the study's own
// reference). The other formats use the same exact-sum oracle (dot8::exact_round) with the same
// quantization rule; for E4M3 and E2M1 that generic path is also checked against mx_chain here.
#include "../datapaths.h"
#include <cstdio>
#include <stdexcept>

struct Spec {
    int exp_bits, man_bits, emax, block;
    double max_value;
    bool nvfp4;
};
constexpr Spec specs[] = {{4, 3, 8, 32, 448.0, false}, {5, 2, 15, 32, 57344.0, false},
                          {3, 2, 4, 32, 28.0, false},  {2, 3, 2, 32, 7.5, false},
                          {2, 1, 2, 32, 6.0, false},   {2, 1, 2, 16, 6.0, true}};

struct Element {
    bool sign;
    uint32_t sig;
    int lsb;
};

static Element decode(uint8_t code, int exp_bits, int man_bits) {
    dot8::Operand x = dot8::decode(code, unsigned(man_bits), unsigned(exp_bits),
                                   (1 << (exp_bits - 1)) - 1);
    return {x.sign, x.sig, x.lsb};
}

// One quantized row: element codes plus each block's scale as sig * 2^lsb.
struct Row {
    std::vector<Element> e;
    std::vector<Element> scale;
    double tensor_decode = 1;
};

// NVFP4 (arXiv 2509.25149, Appendix B) in float arithmetic: encode = 2688 / amax_tensor, block
// scale = E4M3(amax_block / 6 * encode), element = E2M1(x * encode / block scale).
static Row quantize(const float *x, int k, const Spec &s, double tensor_amax) {
    Row r;
    float encode = 2688.0f / float(tensor_amax);
    if (s.nvfp4) r.tensor_decode = tensor_amax / (6.0 * 448.0);
    for (int b = 0; b < k; b += s.block) {
        float m = 0;
        for (int i = 0; i < s.block; ++i) m = std::max(m, std::fabs(x[b + i]));
        Element scale;
        float divide;  // the element is to_small(x * pre / divide)
        float pre = 1;
        if (s.nvfp4) {
            scale = decode(to_small(double(m / 6.0f * encode), 4, 3, 448.0), 4, 3);
            pre = encode;
            divide = std::ldexp(float(scale.sig), scale.lsb);
        } else {
            int shared = m > 0 ? int(std::floor(std::log2(m))) - s.emax : -127;
            shared = std::clamp(shared, -127, 127);
            scale = {false, 1, shared};
            divide = std::ldexp(1.0f, shared);
        }
        r.scale.push_back(scale);
        for (int i = 0; i < s.block; ++i) {
            uint8_t code = divide == 0 ? 0
                                       : to_small(double(x[b + i] * pre / divide), s.exp_bits,
                                                  s.man_bits, s.max_value);
            r.e.push_back(decode(code, s.exp_bits, s.man_bits));
        }
    }
    return r;
}

static double value(const Row &r, int i, int block) {
    const Element &e = r.e[i], &s = r.scale[i / block];
    double v = std::ldexp(double(e.sig) * double(s.sig), e.lsb + s.lsb);
    return e.sign ? -v : v;
}

// Chained exact block dot products, one FP32 rounding per block, from +0.
static float oracle(const Row &a, const Row &b, int k, int block) {
    uint32_t acc = 0;
    for (int b0 = 0; b0 < k; b0 += block) {
        std::vector<dot8::Term> t;
        const Element &sa = a.scale[b0 / block], &sb = b.scale[b0 / block];
        for (int i = b0; i < b0 + block; ++i) {
            const Element &x = a.e[i], &y = b.e[i];
            t.push_back({x.sign != y.sign, x.sig * y.sig * sa.sig * sb.sig,
                         x.lsb + y.lsb + sa.lsb + sb.lsb, 0});
        }
        dot8::Operand z = dot8::decode(acc, 23, 8, 127);
        t.push_back({z.sign, z.sig, z.lsb, 0});
        acc = dot8::exact_round(t.data(), t.data() + t.size(), false, false).bits;
    }
    return value(acc);
}

int main() {
    int32_t head[4];
    if (std::fread(head, 4, 4, stdin) != 4) return 1;
    int f = head[0], m = head[1], n = head[2], k = head[3];
    const Spec &s = specs[f];
    std::vector<float> a(size_t(m) * k), b(size_t(n) * k);
    if (std::fread(a.data(), 4, a.size(), stdin) != a.size()) return 1;
    if (std::fread(b.data(), 4, b.size(), stdin) != b.size()) return 1;
    auto tensor_amax = [](const std::vector<float> &x) {
        double r = 0;
        for (float v : x) r = std::max(r, std::fabs(double(v)));
        return r > 0 ? r : 1.0;
    };
    double amax_a = tensor_amax(a), amax_b = tensor_amax(b);
    std::vector<Row> ra, rb;
    for (int i = 0; i < m; ++i) ra.push_back(quantize(&a[size_t(i) * k], k, s, amax_a));
    for (int j = 0; j < n; ++j) rb.push_back(quantize(&b[size_t(j) * k], k, s, amax_b));
    for (const auto *rows : {&ra, &rb})
        for (const Row &r : *rows)
            for (int i = 0; i < k; ++i) {
                double v = value(r, i, s.block);
                std::fwrite(&v, 8, 1, stdout);
            }
    float decode = float(ra[0].tensor_decode * rb[0].tensor_decode);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float c = oracle(ra[i], rb[j], k, s.block);
            if (f == 0 || f == 4) {
                Vec x(a.begin() + size_t(i) * k, a.begin() + size_t(i + 1) * k);
                Vec y(b.begin() + size_t(j) * k, b.begin() + size_t(j + 1) * k);
                float study = mx_chain(x, y, f == 0 ? MxElement::e4m3 : MxElement::e2m1);
                if (bits(study) != bits(c)) throw std::runtime_error("oracle != mx_chain");
            }
            if (s.nvfp4) c = c * decode;
            std::fwrite(&c, 4, 1, stdout);
        }
    return 0;
}
