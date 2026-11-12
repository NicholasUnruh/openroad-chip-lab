// The numerics study's MXFP8 training run (tools/numerics/train.h) with every GEMM executed by the
// Verilated training_tile_v7. Each tile GEMM is also computed by the study's MXFP8 datapath and
// must match it bit for bit, so the loss curve is the hardware's.
// usage: Vtraining_tile_v7 [steps] [seeds]
#include "mx_tile.h"
#include "../tools/numerics/train.h"

static Chip *chip;
static uint64_t tile_gemms = 0;

// Quantizes K blocks of each A row and B column like the study, pads M and N to whole tiles and
// K to whole blocks, and runs the largest chunks that fit the scratchpad.
static Matrix tile_gemm(const Matrix &a, const Matrix &b, unsigned m, unsigned k, unsigned n) {
    unsigned blocks = (k + 31) / 32, mt_all = (m + 7) / 8, nt_all = (n + 7) / 8;
    require(blocks <= 4, "K above 128 needs a host-side split");
    unsigned budget = 256 / (17 * blocks);  // tiles of A plus B per GEMM
    unsigned chunk_m = std::min(4u, budget - 1), chunk_n = std::min(4u, budget - chunk_m);
    unsigned kp = 32 * blocks;
    std::vector<uint8_t> qa(8 * mt_all * kp, 0), qb(8 * nt_all * kp, 0), sa(8 * mt_all * blocks, 0),
        sb(8 * nt_all * blocks, 0);
    std::array<uint8_t, 32> codes;
    Vec row(kp);
    for (unsigned i = 0; i < m; ++i) {
        std::fill(row.begin(), row.end(), 0.0);
        for (unsigned x = 0; x < k; ++x) row[x] = a[i * k + x];
        for (unsigned kb = 0; kb < blocks; ++kb) {
            sa[i * blocks + kb] = mx_quantize(&row[32 * kb], MxElement::e4m3, codes);
            std::copy(codes.begin(), codes.end(), &qa[i * kp + 32 * kb]);
        }
    }
    for (unsigned j = 0; j < n; ++j) {
        std::fill(row.begin(), row.end(), 0.0);
        for (unsigned x = 0; x < k; ++x) row[x] = b[x * n + j];
        for (unsigned kb = 0; kb < blocks; ++kb) {
            sb[j * blocks + kb] = mx_quantize(&row[32 * kb], MxElement::e4m3, codes);
            std::copy(codes.begin(), codes.end(), &qb[j * kp + 32 * kb]);
        }
    }
    Matrix c(m * n);
    for (unsigned t0 = 0; t0 < mt_all; t0 += chunk_m)
        for (unsigned u0 = 0; u0 < nt_all; u0 += chunk_n) {
            MxGemm g{std::min(chunk_m, mt_all - t0), std::min(chunk_n, nt_all - u0), blocks};
            g.a.assign(&qa[8 * t0 * kp], &qa[8 * (t0 + g.m) * kp]);
            g.b.assign(&qb[8 * u0 * kp], &qb[8 * (u0 + g.n) * kp]);
            g.sa.assign(&sa[8 * t0 * blocks], &sa[8 * (t0 + g.m) * blocks]);
            g.sb.assign(&sb[8 * u0 * blocks], &sb[8 * (u0 + g.n) * blocks]);
            g.pack(0);
            chip->store(g);
            std::vector<uint32_t> out = chip->run(g);
            ++tile_gemms;
            for (unsigned r = 0; r < 8 * g.m; ++r)
                for (unsigned col = 0; col < 8 * g.n; ++col) {
                    unsigned i = 8 * t0 + r, j = 8 * u0 + col;
                    if (i < m && j < n) c[i * n + j] = fp32_value(out[r * 8 * g.n + col]);
                }
        }
    return c;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    unsigned steps = argc > 1 ? unsigned(std::stoul(argv[1])) : 3000;
    unsigned seeds = argc > 2 ? unsigned(std::stoul(argv[2])) : 3;
    try {
        Chip tile;
        chip = &tile;
        GemmFn study = per_dot([](const Vec &a, const Vec &b, double, double) {
            return mx_chain(a, b, MxElement::e4m3);
        });
        GemmFn checked = [&](const Matrix &a, const Matrix &b, unsigned m, unsigned k, unsigned n) {
            Matrix got = tile_gemm(a, b, m, k, n), expected = study(a, b, m, k, n);
            for (size_t i = 0; i < got.size(); ++i)
                require(
                    got[i] == expected[i] || (std::isnan(got[i]) && std::isnan(expected[i])),
                    "tile GEMM differs from the study datapath at element " + std::to_string(i));
            return got;
        };
        print_header();
        train({"mxfp8", checked, checked}, steps, seeds);
        std::cout << "PASS: training_tile_v7 ran " << tile_gemms << " tile GEMMs in " << tile.cycles
                  << " clocks, all bit-exact against the study's MXFP8 datapath\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
