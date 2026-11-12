// C++ side of the randomized cross-check (xcheck.py): quantizes and multiplies FP32 matrices with
// the numerics study's references and writes the results for a bit-for-bit comparison.
//
// stdin:  int32 format, M, N, K; float A[M*K]; float B[N*K]   (C = A B^T, blocks along K)
// stdout: double dequantized A[M*K], B[N*K]; float C[M*N]
// Formats: 0 MXFP8 E4M3, 1 MXFP8 E5M2, 2 MXFP6 E3M2, 3 MXFP6 E2M3, 4 MXFP4 E2M1, 5 NVFP4.
//
// Every GEMM output is computed twice and required to agree: by the study's references (mx_chain:
// mx_quantize plus mx_dot32; nvfp4_chain: nvfp4_quantize plus nvfp4_dot32 and the host-side
// tensor decode) and by a generic exact-sum oracle (dot8::exact_round, one rounding per 32
// elements). E5M2 has no study chain here and uses the oracle alone.
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
};

// MX rows follow the OCP shared-exponent rule; NVFP4 rows come from the study's nvfp4_quantize.
static Row quantize(const float *x, int k, const Spec &s, double tensor_amax) {
    Row r;
    if (s.nvfp4) {
        float s_enc = nvfp4_encode_scale(tensor_amax);
        for (int b = 0; b < k; b += 16) {
            double block[16];
            for (int i = 0; i < 16; ++i) block[i] = x[b + i];
            std::array<uint8_t, 16> codes;
            r.scale.push_back(decode(nvfp4_quantize(block, s_enc, codes), 4, 3));
            for (uint8_t c : codes) r.e.push_back(decode(c, 2, 1));
        }
        return r;
    }
    for (int b = 0; b < k; b += s.block) {
        float m = 0;
        for (int i = 0; i < s.block; ++i) m = std::max(m, std::fabs(x[b + i]));
        int shared = m > 0 ? int(std::floor(std::log2(m))) - s.emax : -127;
        shared = std::clamp(shared, -127, 127);
        r.scale.push_back({false, 1, shared});
        float divide = std::ldexp(1.0f, shared);
        for (int i = 0; i < s.block; ++i) {
            uint8_t code = to_small(double(x[b + i] / divide), s.exp_bits, s.man_bits, s.max_value);
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

// Chained exact dot products, one FP32 rounding per 32 elements, from +0.
static float oracle(const Row &a, const Row &b, int k, int block) {
    uint32_t acc = 0;
    for (int b0 = 0; b0 < k; b0 += 32) {  // one unit call (one or two scale blocks) per 32
        std::vector<dot8::Term> t;
        for (int i = b0; i < b0 + 32; ++i) {
            const Element &x = a.e[i], &y = b.e[i];
            const Element &sa = a.scale[i / block], &sb = b.scale[i / block];
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
    constexpr MxElement element[] = {MxElement::e4m3, MxElement::e5m2, MxElement::e3m2,
                                     MxElement::e2m3, MxElement::e2m1};
    float decode = (1.0f / nvfp4_encode_scale(amax_a)) * (1.0f / nvfp4_encode_scale(amax_b));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            float c = oracle(ra[i], rb[j], k, s.block);
            if (s.nvfp4) c = c * decode;
            if (f != 1) {
                Vec x(a.begin() + size_t(i) * k, a.begin() + size_t(i + 1) * k);
                Vec y(b.begin() + size_t(j) * k, b.begin() + size_t(j + 1) * k);
                float study = s.nvfp4 ? nvfp4_chain(x, y, amax_a, amax_b) : mx_chain(x, y, element[f]);
                if (bits(study) != bits(c)) throw std::runtime_error("oracle != study chain");
            }
            std::fwrite(&c, 4, 1, stdout);
        }
    return 0;
}
