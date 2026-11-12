// Stochastic rounding check (docs/numerics-reference.md): E2M1 SR must be unbiased per element, and
// prints how much an SR gradient operand spreads an MXFP4 dot product.
// usage: build/sr_check > results/numerics/sr_check.txt
#include "datapaths.h"
#include <cstdio>
#include <memory>
#include <random>

int main() {
    auto generator = std::make_shared<std::mt19937>(20261111);
    std::function<uint16_t()> next = [generator] { return uint16_t((*generator)() >> 16); };
    constexpr int draws = 200000;
    std::printf("E2M1 stochastic rounding, mean of %d draws\nvalue mean\n", draws);
    for (double x : {0.1, 0.3, 0.74, 1.2, 2.6, 3.3, 4.7, 5.5, 5.99}) {
        double sum = 0;
        for (int i = 0; i < draws; ++i) {
            dot8::Operand q = dot8::decode(to_small(x, 2, 1, 6.0, next()), 1, 2, 1);
            sum += std::ldexp(double(q.sig), q.lsb);
        }
        std::printf("%.3f %.4f\n", x, sum / draws);
    }
    // 64-term dot products: a ~ N(0, 1) rounded to nearest, b heavy-tailed (N(0, 1) * e^N(0, 1),
    // gradient-like) rounded stochastically.
    std::mt19937_64 rng(1);
    std::normal_distribution<double> normal(0, 1);
    constexpr int repeats = 20000;
    std::printf("\nMXFP4 64-term dot products, SR on b, %d repeats\nexact rne sr_mean sr_std\n",
                repeats);
    for (int trial = 0; trial < 4; ++trial) {
        Vec a(64), b(64);
        for (double &v : a) v = normal(rng);
        for (double &v : b) v = normal(rng) * std::exp(normal(rng));
        double exact = 0, sum = 0, square = 0;
        for (int i = 0; i < 64; ++i) exact += a[i] * b[i];
        for (int i = 0; i < repeats; ++i) {
            double v = mx_chain_sr(a, b, MxElement::e2m1, next);
            sum += v;
            square += v * v;
        }
        double mean = sum / repeats;
        std::printf("%.4f %.4f %.4f %.4f\n", exact, mx_chain(a, b, MxElement::e2m1), mean,
                    std::sqrt(square / repeats - mean * mean));
    }
    return 0;
}
