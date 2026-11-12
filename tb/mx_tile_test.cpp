// GEMM regression for training_tile_v7: every result bit-exact against tb/mx_tile.h's reference.
#include "mx_tile.h"
#include <iostream>

constexpr const char *design_name = "training_tile_v7";
constexpr double target_mhz = 40;

// E4M3 codes of 0, +-0.5, +-1, +-1.5, +-2 and +-3: with scale 2^0 every block sum is exact.
static uint8_t exact_code(std::mt19937 &rng) {
    static const uint8_t codes[] = {0x00, 0x30, 0x38, 0x3c, 0x40, 0x44};
    return uint8_t(codes[rng() % 6] | (rng() & 1 ? 0x80 : 0));
}

// kind 0: exact values and unit scales; 1: random finite codes with scales 2^-10 .. 2^10;
// 2: raw bytes for elements and scales (NaN codes, NaN scales, extreme scales).
static MxGemm random_gemm(std::mt19937 &rng, unsigned m, unsigned n, unsigned k, unsigned kind) {
    MxGemm g{m, n, k};
    auto element = [&]() -> uint8_t {
        if (kind == 0) return exact_code(rng);
        uint8_t x = uint8_t(rng());
        return kind == 1 && (x & 0x7f) == 0x7f ? uint8_t(x - 1) : x;
    };
    auto scale = [&]() -> uint8_t {
        return kind == 0 ? 127 : kind == 1 ? uint8_t(117 + rng() % 21) : uint8_t(rng());
    };
    g.a.resize(8 * m * g.inner());
    g.b.resize(8 * n * g.inner());
    g.sa.resize(8 * m * k);
    g.sb.resize(8 * n * k);
    for (uint8_t &x : g.a) x = element();
    for (uint8_t &x : g.b) x = element();
    for (uint8_t &x : g.sa) x = scale();
    for (uint8_t &x : g.sb) x = scale();
    g.pack(rng() % 256);  // exercises row wraparound
    return g;
}

static void shape_tests(Chip &chip) {
    std::mt19937 rng(5150);
    unsigned count = 0;
    for (unsigned m = 1; m <= 4; ++m)
        for (unsigned n = 1; n <= 4; ++n)
            for (unsigned k = 1; k <= 4; ++k) {
                if (17 * (m + n) * k > 256) continue;  // operands and scales must fit
                for (unsigned kind = 0; kind < 2; ++kind) {
                    MxGemm g = random_gemm(rng, m, n, k, kind);
                    HostPolicy policy{count % 2 ? 30u : 0u, count % 3 ? 0u : 150u};
                    chip.check(g, policy,
                               "shape GEMM m=" + std::to_string(m) + " n=" + std::to_string(n) +
                                   " k=" + std::to_string(k) + " kind=" + std::to_string(kind));
                    if (kind == 0)
                        require(chip.dut.exception_flags == 0, "exact GEMM raises no flags");
                    ++count;
                }
            }
    unsigned raw = 0;
    for (; raw < 64; ++raw) {
        unsigned m = 1 + rng() % 4, n = 1 + rng() % 4, k = 1 + rng() % 4;
        while (17 * (m + n) * k > 256) k = k - 1;
        chip.check(random_gemm(rng, m, n, k, 2), {}, "raw-encoding GEMM");
    }
    std::cout << "PASS: " << count
              << " GEMM shapes (1-4 tiles of M and N, 1-4 K blocks, all that fit), exact and "
                 "scaled; "
              << raw
              << " raw E4M3/E8M0 encoding GEMMs; wraparound bases; stalled and delayed drains\n";
}

static void flag_tests(Chip &chip) {
    std::mt19937 rng(77);
    MxGemm g = random_gemm(rng, 2, 3, 2, 0);
    g.a[5 * g.inner() + 40] = 0x7f;  // E4M3 NaN: row 5 is NaN, quietly
    std::vector<uint32_t> out = chip.check(g, {}, "NaN element GEMM");
    require(chip.dut.exception_flags == 0 && out[5 * 24] == 0x7fc00000u, "E4M3 NaN is quiet");
    g = random_gemm(rng, 1, 2, 1, 0);
    g.sb[3] = 0xff;  // E8M0 NaN scale: column 3 is NaN
    out = chip.check(g, {}, "NaN scale GEMM");
    require(chip.dut.exception_flags == 0 && out[3] == 0x7fc00000u, "NaN scale is quiet");
    g = random_gemm(rng, 1, 1, 3, 0);
    for (uint8_t &s : g.sa) s = 254;
    for (uint8_t &s : g.sb) s = 254;
    for (uint8_t &x : g.a) x = 0x40;
    for (uint8_t &x : g.b) x = 0x40;
    chip.check(g, {}, "overflow GEMM");
    require(chip.dut.exception_flags == 5, "overflow raises overflow and inexact");
    g = random_gemm(rng, 1, 1, 1, 0);
    for (uint8_t &s : g.sa) s = 1;  // 2^-126 * 2^-126: every nonzero block sum rounds to zero
    for (uint8_t &s : g.sb) s = 1;
    g.a[0] = 0x3c;
    chip.check(g, {}, "underflow GEMM");
    require(chip.dut.exception_flags == 3, "tiny inexact result raises underflow and inexact");
    chip.check(random_gemm(rng, 1, 2, 1, 0), {}, "flags cleared GEMM");
    require(chip.dut.exception_flags == 0, "flags clear on start");
    std::cout << "PASS: quiet E4M3 NaN and NaN scale; overflow and underflow flags combine across "
                 "jobs and clear on start\n";
}

static void protocol_tests(Chip &chip) {
    std::mt19937 rng(31337);
    MxGemm g = random_gemm(rng, 3, 3, 1, 1);
    chip.store(g);
    compare(chip.run(g, {}, 5), reference(g), "start ignored while busy");
    compare(chip.run(g, {}, 300), reference(g), "late start ignored while busy");
    unsigned aborts = 0;
    for (int abort : {0, 1, 20, 34, 35, 60, 140, 400, 800, 1100}) {
        chip.run(g, {0, 300}, -1, abort);
        compare(chip.run(g), reference(g), "restart after abort at " + std::to_string(abort));
        ++aborts;
    }
    // Overlap: write the next GEMM's operands into free rows while the current one runs.
    MxGemm next = random_gemm(rng, 2, 2, 1, 1);
    next.pack(g.a_base + g.sp_rows());
    chip.writes = Chip::operand_writes(next);
    compare(chip.run(g, {20, 10}), reference(g), "GEMM during scratchpad writes");
    require(chip.writes.empty(), "overlapped writes finish");
    compare(chip.run(next), reference(next), "GEMM from overlapped writes");
    std::cout << "PASS: busy start rejection, " << aborts
              << " reset aborts and restarts, scratchpad writes overlapped with computation\n";
}

static void benchmarks(Chip &chip) {
    std::mt19937 rng(8664);
    // One 32x32x32 GEMM: the host writes A, B and scales, then drains every output tile.
    MxGemm big = random_gemm(rng, 4, 4, 1, 1);
    big.pack(0);
    uint64_t begin = chip.cycles, begin_bytes = chip.written_bytes + chip.read_bytes;
    chip.check(big, {}, "GEMM32 benchmark");
    uint64_t cycles = chip.cycles - begin,
             bytes = chip.written_bytes + chip.read_bytes - begin_bytes;
    unsigned flops = 2 * 32 * 32 * 32, fill = 2 * 32 * 32 / 4 + 2 * 4 * 2;
    require(bytes == 4 * fill + 32 * 32 * 4, "GEMM32 traffic accounting");
    // After the fill, the 1,024-word FP32 drain bounds the run; 16 jobs at 34 clocks fit under it.
    require(cycles - fill >= 1024 && cycles - fill <= 1024 + 120, "GEMM32 drain-bound throughput");
    std::cout << "BENCH design=" << design_name << " mode=gemm32_scratchpad cycles=" << cycles
              << " bytes=" << bytes << " flops=" << flops
              << " target_mflops=" << double(flops) / cycles * target_mhz << '\n';

    // Eight 16x16x32 GEMMs: the next operands load into the other scratchpad half while one runs.
    constexpr unsigned count = 8;
    std::vector<MxGemm> gemms;
    for (unsigned n = 0; n < count; ++n) {
        MxGemm g = random_gemm(rng, 2, 2, 1, 1);
        g.pack((n & 1) * 128);
        gemms.push_back(g);
    }
    begin = chip.cycles;
    begin_bytes = chip.written_bytes + chip.read_bytes;
    chip.store(gemms[0]);
    for (unsigned n = 0; n < count; ++n) {
        if (n + 1 < count) chip.writes = Chip::operand_writes(gemms[n + 1]);
        compare(chip.run(gemms[n]), reference(gemms[n]), "GEMM16 pipelined benchmark");
        for (unsigned limit = 0; chip.drive_write(); ++limit) {
            require(limit < 1024, "write drain bound");
            chip.tick();
        }
        chip.dut.sp_write = 0;
    }
    cycles = chip.cycles - begin;
    bytes = chip.written_bytes + chip.read_bytes - begin_bytes;
    flops = count * 2 * 16 * 16 * 32;
    require(bytes == count * (2 * 16 * 32 + 2 * 2 * 8 + 16 * 16 * 4), "GEMM16 traffic accounting");
    std::cout << "BENCH design=" << design_name << " mode=gemm16x32_pipelined matrices=" << count
              << " cycles=" << cycles << " bytes=" << bytes << " flops=" << flops
              << " target_mflops=" << double(flops) / cycles * target_mhz << '\n';

    // Compute-bound: one 16x16x128 GEMM (four K blocks), drained only 1 KB for 16 jobs.
    MxGemm deep = random_gemm(rng, 2, 1, 4, 1);
    deep.pack(0);
    chip.store(deep);
    begin = chip.cycles;
    chip.run(deep);
    cycles = chip.cycles - begin;
    unsigned jobs = deep.m * deep.n * deep.k;
    // Back-to-back loads give 34 clocks per job, plus 94: the first job starts at load index 20,
    // then the 7-clock unit latency and the last tile's 64-word drain. One lost clock per job
    // would add `jobs`.
    require(cycles >= 34 * jobs && cycles <= 34 * jobs + 94, "compute-bound job rate");
    std::cout << "BENCH design=" << design_name << " mode=gemm16x8x128_compute jobs=" << jobs
              << " cycles=" << cycles << " macs=" << 64 * jobs * 32 << '\n';
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    try {
        Chip chip;
        shape_tests(chip);
        flag_tests(chip);
        protocol_tests(chip);
        benchmarks(chip);
        chip.dut.final();
        std::cout << "PASS: " << design_name << " gemms=" << chip.gemms << " jobs=" << chip.jobs
                  << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
