// Training emulation of every numerics-study recipe (tools/numerics/train.h).
// usage: build/train_mlp [steps] [seeds] [recipe]
#include "train.h"

static std::vector<Recipe> recipes() {
    Dot fp32 = [](const Vec &a, const Vec &b, double, double) { return fp32_dot(a, b); };
    Dot bf16_fma_path = [](const Vec &a, const Vec &b, double, double) { return bf16_fma(a, b); };
    Dot bf16 = [](const Vec &a, const Vec &b, double, double) {
        return dot8_chain(a, b, dot8_bf16);
    };
    auto frame = [](int bits) {
        return per_dot([bits](const Vec &a, const Vec &b, double, double) {
            return dot8_chain(a, b, dot8_bf16, 0, 0, bits);
        });
    };
    Dot e4m3 = [](const Vec &a, const Vec &b, double x, double y) {
        return dot8_chain(a, b, dot8_e4m3, x, y);
    };
    Dot e5m2 = [](const Vec &a, const Vec &b, double x, double y) {
        return dot8_chain(a, b, dot8_e5m2, x, y);
    };
    Dot mxfp8 = [](const Vec &a, const Vec &b, double, double) {
        return mx_chain(a, b, MxElement::e4m3);
    };
    Dot mxfp4 = [](const Vec &a, const Vec &b, double, double) {
        return mx_chain(a, b, MxElement::e2m1);
    };
    return {{"fp32", per_dot(fp32), per_dot(fp32)},
            {"bf16_fma", per_dot(bf16_fma_path), per_dot(bf16_fma_path)},
            {"bf16_dot8", per_dot(bf16), per_dot(bf16)},
            {"bf16_dot8_f24", frame(24), frame(24)},
            {"bf16_dot8_f28", frame(28), frame(28)},
            {"bf16_dot8_f36", frame(36), frame(36)},
            {"bf16_dot8_f40", frame(40), frame(40)},
            {"fp8_tensor", per_dot(e4m3), per_dot(e5m2)},
            {"mxfp8", per_dot(mxfp8), per_dot(mxfp8)},
            {"mxfp4_fwd_mxfp8_bwd", per_dot(mxfp4), per_dot(mxfp8)},
            {"mxfp4", per_dot(mxfp4), per_dot(mxfp4)}};
}

int main(int argc, char **argv) {
    unsigned steps = argc > 1 ? unsigned(std::stoul(argv[1])) : 1500;
    unsigned seeds = argc > 2 ? unsigned(std::stoul(argv[2])) : 3;
    std::string only = argc > 3 ? argv[3] : "";
    print_header();
    for (const Recipe &r : recipes())
        if (only.empty() || r.name == only) train(r, steps, seeds);
    return 0;
}
