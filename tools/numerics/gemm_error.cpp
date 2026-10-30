// Dot-product error of each study datapath against FP64 on the original, unquantized data.
// Error is |y - y64| / sum|a_i b_i|, which stays meaningful under cancellation.
// usage: build/gemm_error [samples]
#include "datapaths.h"
#include <functional>
#include <iostream>
#include <random>
#include <string>

int main(int argc, char **argv) {
    unsigned samples = argc > 1 ? unsigned(std::stoul(argv[1])) : 2000;
    // Tensor-scaled paths share one amax across all sample vectors of an operand, as per-tensor
    // FP8/INT8 training recipes do; row-scaled paths use each vector's own amax.
    double tensor_a = 1, tensor_b = 1;
    using Path = std::function<float(const Vec &, const Vec &)>;
    std::vector<std::pair<std::string, Path>> paths = {
        {"fp32_fma", fp32_dot},
        {"bf16_fma", bf16_fma},
        {"bf16_dot8", [](const Vec &a, const Vec &b) { return dot8_chain(a, b, dot8_bf16); }},
        {"e4m3_tensor", [&](const Vec &a, const Vec &b) { return dot8_chain(a, b, dot8_e4m3, tensor_a, tensor_b); }},
        {"e5m2_tensor", [&](const Vec &a, const Vec &b) { return dot8_chain(a, b, dot8_e5m2, tensor_a, tensor_b); }},
        {"e4m3_row", [](const Vec &a, const Vec &b) { return dot8_chain(a, b, dot8_e4m3); }},
        {"mxfp8", [](const Vec &a, const Vec &b) { return mx_chain(a, b, MxElement::e4m3); }},
        {"mxfp4", [](const Vec &a, const Vec &b) { return mx_chain(a, b, MxElement::e2m1); }},
        {"int8_tensor", [&](const Vec &a, const Vec &b) { return int8_dot(a, b, tensor_a, tensor_b); }},
        {"int8_row", [](const Vec &a, const Vec &b) { return int8_dot(a, b); }},
    };
    std::cout << "distribution K datapath median_err p99_err\n";
    for (const std::string dist : {"gaussian", "laplace", "outliers", "rows"}) {
        for (unsigned K : {256u, 4096u}) {
            std::mt19937_64 rng(K * 7919 + dist.size());
            std::normal_distribution<double> normal(0, 1);
            std::exponential_distribution<double> expo(1);
            std::vector<std::vector<double>> errors(paths.size());
            std::vector<Vec> as(samples, Vec(K)), bs(samples, Vec(K));
            tensor_a = tensor_b = 0;
            for (unsigned s = 0; s < samples; ++s) {
                // "rows" gives each vector a log-normal magnitude, like activations of different tokens.
                double row_scale = dist == "rows" ? std::exp(1.5 * normal(rng)) : 1;
                for (unsigned i = 0; i < K; ++i) {
                    if (dist == "laplace") {
                        as[s][i] = (rng() & 1 ? 1 : -1) * expo(rng);
                        bs[s][i] = (rng() & 1 ? 1 : -1) * expo(rng);
                    } else {
                        as[s][i] = normal(rng) * row_scale;
                        bs[s][i] = normal(rng);
                        if (dist == "outliers" && rng() % 100 == 0) as[s][i] *= 100;
                    }
                }
                tensor_a = std::max(tensor_a, amax(as[s]));
                tensor_b = std::max(tensor_b, amax(bs[s]));
            }
            for (unsigned s = 0; s < samples; ++s) {
                const Vec &a = as[s], &b = bs[s];
                double exact = 0, magnitude = 0;
                for (unsigned i = 0; i < K; ++i) {
                    exact += a[i] * b[i];
                    magnitude += std::fabs(a[i] * b[i]);
                }
                for (size_t p = 0; p < paths.size(); ++p)
                    errors[p].push_back(std::fabs(double(paths[p].second(a, b)) - exact) / magnitude);
            }
            for (size_t p = 0; p < paths.size(); ++p) {
                auto &e = errors[p];
                std::sort(e.begin(), e.end());
                std::cout << dist << ' ' << K << ' ' << paths[p].first << ' ' << e[e.size() / 2] << ' '
                          << e[e.size() * 99 / 100] << '\n';
            }
        }
    }
    return 0;
}
