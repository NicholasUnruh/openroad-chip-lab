#ifdef TILE_V6
#include "Vtraining_tile_v6.h"
#include "dot8_model.h"
using Model = Vtraining_tile_v6;
constexpr const char *design_name = "training_tile_v6";
#else
#include "Vtraining_tile_v5.h"
using Model = Vtraining_tile_v5;
constexpr const char *design_name = "training_tile_v5";
#endif
#include "verilated.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

constexpr double target_mhz = 40;
using Matrix = std::vector<float>;

static uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

static float value(uint32_t raw) {
    float result;
    std::memcpy(&result, &raw, sizeof(result));
    return result;
}

static float bf16(float x) {
    uint32_t raw = bits(x);
    if ((raw & 0xffffu) == 0) return x;
    if ((raw & 0x7f800000u) == 0x7f800000u && (raw & 0x7fffffu))
        return value((raw & 0xffff0000u) | 0x00400000u);
    return value((raw + 0x7fffu + ((raw >> 16) & 1u)) & 0xffff0000u);
}

static void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

// Logical GEMM dimensions in 8x8 tiles; operands are stored transposed when requested.
struct Gemm {
    unsigned m, n, k;
    bool ta = false, tb = false;
    unsigned a_base = 0, b_base = 0;
    Matrix a, b;  // stored row-major, BF16-representable
    unsigned a_rows() const { return 8 * (ta ? k : m); }
    unsigned a_cols() const { return 8 * (ta ? m : k); }
    unsigned b_rows() const { return 8 * (tb ? n : k); }
    unsigned b_cols() const { return 8 * (tb ? k : n); }
    unsigned a_sp_rows() const { return a_rows() * a_cols() / 8; }
    unsigned b_sp_rows() const { return b_rows() * b_cols() / 8; }
};

// V5 chains FP32 FMAs in K order; V6 rounds once per 8-term dot product, starting from +0.
static Matrix reference(const Gemm &g) {
    unsigned rows = 8 * g.m, cols = 8 * g.n, inner = 8 * g.k;
    Matrix c(rows * cols, 0.0f);
    for (unsigned i = 0; i < rows; ++i)
        for (unsigned j = 0; j < cols; ++j) {
#ifdef TILE_V6
            uint32_t acc = 0;
            for (unsigned x0 = 0; x0 < inner; x0 += 8) {
                std::array<uint16_t, 8> a8, b8;
                for (unsigned x = 0; x < 8; ++x) {
                    a8[x] = bits(g.ta ? g.a[(x0 + x) * g.a_cols() + i] : g.a[i * g.a_cols() + x0 + x]) >> 16;
                    b8[x] = bits(g.tb ? g.b[j * g.b_cols() + x0 + x] : g.b[(x0 + x) * g.b_cols() + j]) >> 16;
                }
                acc = dot8_spec(a8, b8, acc).bits;
            }
            c[i * cols + j] = value(acc);
#else
            for (unsigned x = 0; x < inner; ++x) {
                float a = g.ta ? g.a[x * g.a_cols() + i] : g.a[i * g.a_cols() + x];
                float b = g.tb ? g.b[j * g.b_cols() + x] : g.b[x * g.b_cols() + j];
                c[i * cols + j] = std::fma(a, b, c[i * cols + j]);
            }
#endif
        }
    return c;
}

static void compare(const Matrix &got, const Matrix &expected, const std::string &label) {
    for (size_t i = 0; i < got.size(); ++i) {
        // HardFloat's RISC-V specialization returns canonical NaNs.
        bool equal = std::isnan(expected[i]) ? bits(got[i]) == 0x7fc00000u
                                           : bits(got[i]) == bits(expected[i]);
        require(equal, label + " element=" + std::to_string(i) +
                       " got_bits=" + std::to_string(bits(got[i])) +
                       " expected_bits=" + std::to_string(bits(expected[i])));
    }
}

// Host behaviour while draining: probability of skipping a read cycle and maximum ack delay.
struct HostPolicy { unsigned stall_percent = 0, max_ack_delay = 0; };

class Chip {
public:
    Model dut;
    std::mt19937 rng{20261018};
    uint64_t cycles = 0, written_bytes = 0, read_bytes = 0;
    unsigned gemms = 0, jobs = 0;
    std::deque<std::pair<unsigned, uint32_t>> writes;  // overlapped scratchpad writes

    Chip() { reset(); }

    void tick() {
        ++cycles;
        dut.clk = 0;
        dut.eval();
        dut.clk = 1;
        dut.eval();
        dut.clk = 0;
        dut.eval();
    }

    void idle_inputs() {
        dut.sp_write = 0;
        dut.gemm_start = 0;
        dut.result_ack = 0;
    }

    void reset() {
        idle_inputs();
        dut.rst = 1;
        tick();
        require(!dut.gemm_busy && !dut.gemm_done && !dut.result_valid && dut.exception_flags == 0,
                "reset control");
        dut.rst = 0;
    }

    // Row-major packing places each aligned 8-element row segment in one scratchpad row.
    static void queue_matrix(std::deque<std::pair<unsigned, uint32_t>> &out, const Matrix &m, unsigned base) {
        for (unsigned i = 0; i < m.size(); i += 2) {
            uint32_t word = (bits(m[i]) >> 16) | (bits(m[i + 1]) & 0xffff0000u);
            out.emplace_back((4 * base + i / 2) & 0x3ffu, word);
        }
    }

    static std::deque<std::pair<unsigned, uint32_t>> operand_writes(const Gemm &g) {
        std::deque<std::pair<unsigned, uint32_t>> out;
        queue_matrix(out, g.a, g.a_base);
        queue_matrix(out, g.b, g.b_base);
        return out;
    }

    bool drive_write() {
        if (writes.empty()) {
            dut.sp_write = 0;
            return false;
        }
        dut.sp_write = 1;
        dut.sp_addr = writes.front().first;
        dut.sp_data = writes.front().second;
        writes.pop_front();
        written_bytes += 4;
        return true;
    }

    void store(const Gemm &g) {
        writes = operand_writes(g);
        while (drive_write()) tick();
        dut.sp_write = 0;
    }

    void command(const Gemm &g) {
        dut.m_tiles = g.m - 1;
        dut.n_tiles = g.n - 1;
        dut.k_tiles = g.k - 1;
        dut.a_base = g.a_base;
        dut.b_base = g.b_base;
        dut.transpose_a = g.ta;
        dut.transpose_b = g.tb;
    }

    // Runs one accepted GEMM to completion, draining tiles under the host policy.
    // hijack_cycle issues a conflicting start while busy; abort_cycle resets mid-run.
    Matrix run(const Gemm &g, HostPolicy policy = {}, int hijack_cycle = -1, int abort_cycle = -1) {
        command(g);
        dut.gemm_start = 1;
        tick();
        dut.gemm_start = 0;
        require(dut.gemm_busy && !dut.gemm_done, "GEMM accept");
        Matrix out(64 * g.m * g.n);
        unsigned tiles = g.m * g.n, next_tile = 0, count = 0, ack_wait = 0;
        int current = -1;
        for (unsigned cycle = 0;; ++cycle) {
            require(cycle < 200000, "GEMM timeout");
            drive_write();
            if (int(cycle) == hijack_cycle) {
                Gemm other = g;
                other.m = 1 + (g.m % 4);
                other.ta = !g.ta;
                other.a_base = g.a_base + 17;
                command(other);
                dut.gemm_start = 1;
            }
            if (int(cycle) == abort_cycle) {
                dut.rst = 1;
                tick();
                dut.rst = 0;
                idle_inputs();
                require(!dut.gemm_busy && !dut.result_valid && !dut.gemm_done, "abort clears GEMM");
                return {};
            }
            dut.result_ack = 0;
            dut.eval();
            if (dut.result_valid) {
                if (current < 0) {
                    current = dut.result_tile;
                    unsigned expected = (next_tile / g.n) << 2 | (next_tile % g.n);
                    require(unsigned(current) == expected, "tile order");
                    ack_wait = policy.max_ack_delay ? rng() % (policy.max_ack_delay + 1) : 0;
                }
                require(dut.result_tile == unsigned(current), "tile stable during drain");
                if (count < 64 && rng() % 100 >= policy.stall_percent) {
                    dut.read_addr = count;
                    dut.eval();
                    unsigned row = 8 * (current >> 2) + count / 8, col = 8 * (current & 3) + count % 8;
                    out[row * 8 * g.n + col] = value(dut.read_data);
                    ++count;
                    read_bytes += 4;
                }
                if (count == 64) {
                    if (ack_wait) --ack_wait;
                    else {
                        dut.result_ack = 1;
                        ++next_tile;
                        count = 0;
                        current = -1;
                    }
                }
            }
            tick();
            dut.gemm_start = 0;
            if (dut.gemm_done) break;
        }
        require(next_tile == tiles && !dut.gemm_busy && !dut.result_valid, "GEMM completion");
        tick();
        require(!dut.gemm_done, "done must pulse once");
        ++gemms;
        jobs += g.m * g.n * g.k;
        return out;
    }

    Matrix check(const Gemm &g, HostPolicy policy = {}, const std::string &label = "GEMM") {
        store(g);
        Matrix out = run(g, policy);
        compare(out, reference(g), label);
        return out;
    }
};

static Matrix random_matrix(std::mt19937 &rng, unsigned size, bool raw) {
    Matrix m(size);
    for (float &x : m) x = raw ? value(rng() & 0xffff0000u) : bf16(float(int(rng() % 251) - 125) / 8);
    return m;
}

static Gemm random_gemm(std::mt19937 &rng, unsigned m, unsigned n, unsigned k, bool ta, bool tb, bool raw) {
    Gemm g{m, n, k, ta, tb};
    g.a = random_matrix(rng, g.a_rows() * g.a_cols(), raw);
    g.b = random_matrix(rng, g.b_rows() * g.b_cols(), raw);
    g.a_base = rng() % 256;
    g.b_base = (g.a_base + g.a_sp_rows()) % 256;  // exercises row wraparound
    return g;
}

static void shape_tests(Chip &chip) {
    std::mt19937 rng(5150);
    unsigned count = 0;
    for (unsigned mode = 0; mode < 4; ++mode)
        for (unsigned m = 1; m <= 4; ++m)
            for (unsigned n = 1; n <= 4; ++n)
                for (unsigned k = 1; k <= 4; ++k) {
                    Gemm g = random_gemm(rng, m, n, k, mode & 1, mode & 2, false);
                    HostPolicy policy{count % 2 ? 30u : 0u, count % 3 ? 0u : 150u};
                    chip.check(g, policy, "shape GEMM m=" + std::to_string(m) + " n=" + std::to_string(n) +
                                          " k=" + std::to_string(k) + " mode=" + std::to_string(mode));
                    require(chip.dut.exception_flags == 0, "exact integer GEMM raises no flags");
                    ++count;
                }
    for (unsigned test = 0; test < 64; ++test) {
        Gemm g = random_gemm(rng, 1 + rng() % 4, 1 + rng() % 4, 1 + rng() % 4, rng() & 1, rng() & 1, true);
        chip.check(g, {}, "raw-encoding GEMM");
    }
    std::cout << "PASS: " << count << " GEMM shapes 1-4 tiles per dimension in all transpose modes; "
              << "64 raw BF16 encoding GEMMs; wraparound bases; stalled and delayed drains\n";
}

static void flag_tests(Chip &chip) {
    std::mt19937 rng(77);
    Gemm g = random_gemm(rng, 2, 3, 2, false, false, false);
    g.a[5 * g.a_cols() + 9] = INFINITY;
    for (unsigned j = 0; j < g.b_cols(); ++j) g.b[9 * g.b_cols() + j] = 0.0f;
    chip.check(g, {}, "invalid GEMM");
    require(chip.dut.exception_flags == 16, "inf*0 raises only invalid");
    g = random_gemm(rng, 1, 1, 3, false, true, false);
    g.a[0] = value(0x7f7f0000u);
    for (unsigned x = 0; x < 24; ++x) g.b[x] = 4.0f;
    chip.check(g, {}, "overflow GEMM");
    require(chip.dut.exception_flags == 5, "overflow raises overflow and inexact");
    chip.check(random_gemm(rng, 1, 2, 1, false, false, false), {}, "flags cleared GEMM");
    require(chip.dut.exception_flags == 0, "flags clear on start");
    std::cout << "PASS: invalid and overflow flags combine across jobs and clear on start\n";
}

static void protocol_tests(Chip &chip) {
    std::mt19937 rng(31337);
    Gemm g = random_gemm(rng, 3, 3, 2, true, false, false);
    chip.store(g);
    compare(chip.run(g, {}, 5), reference(g), "start ignored while busy");
    compare(chip.run(g, {}, 300), reference(g), "late start ignored while busy");
    unsigned aborts = 0;
    for (int abort : {0, 1, 40, 66, 67, 130, 140, 400, 800, 1100}) {
        chip.run(g, {0, 300}, -1, abort);
        compare(chip.run(g), reference(g), "restart after abort at " + std::to_string(abort));
        ++aborts;
    }
    // Overlap: write the next GEMM's operands into free rows while the current one runs.
    Gemm next = random_gemm(rng, 2, 2, 2, false, true, false);
    next.a_base = (g.b_base + g.b_sp_rows()) % 256;
    next.b_base = (next.a_base + next.a_sp_rows()) % 256;
    chip.writes = Chip::operand_writes(next);
    compare(chip.run(g, {20, 10}), reference(g), "GEMM during scratchpad writes");
    require(chip.writes.empty(), "overlapped writes finish");
    compare(chip.run(next), reference(next), "GEMM from overlapped writes");
    std::cout << "PASS: busy start rejection, " << aborts << " reset aborts and restarts, "
                 "scratchpad writes overlapped with computation\n";
}

static double loss(const Matrix &prediction, const Matrix &target) {
    double sum = 0;
    for (size_t i = 0; i < prediction.size(); ++i) {
        double diff = double(prediction[i]) - target[i];
        sum += 0.5 * diff * diff;
    }
    return sum;
}

static Matrix quantize(Matrix m) {
    for (float &x : m) x = bf16(x);
    return m;
}

// Dense layer Y = X W on the chip: forward, dW = X^T dY, dX = dY W^T; FP32 SGD on the host.
static void training_test(Chip &chip) {
    constexpr unsigned size = 16;
    Matrix x(size * size, 0.0f), target_w(size * size), w(size * size, 0.0f);
    for (unsigned i = 0; i < size; ++i) {
        x[i * size + i] = 1;
        x[i * size + (i + 3) % size] = (i & 1) ? 0.25f : -0.25f;
    }
    for (unsigned i = 0; i < size * size; ++i) target_w[i] = float(int(i % 15) - 7) / 8;
    auto gemm = [](const Matrix &a, const Matrix &b, bool ta, bool tb, unsigned base) {
        Gemm g{2, 2, 2, ta, tb, base, (base + 32) % 256, quantize(a), quantize(b)};
        return g;
    };
    Gemm forward_target = gemm(x, target_w, false, false, 0);
    Matrix target = reference(forward_target);
    double initial = 0, final_loss = 0;
    constexpr unsigned epochs = 40;
    for (unsigned epoch = 0; epoch <= epochs; ++epoch) {
        Matrix y = chip.check(gemm(x, w, false, false, 64), {}, "training forward");
        final_loss = loss(y, target);
        if (epoch == 0) initial = final_loss;
        if (epoch == epochs) break;
        Matrix dy(size * size);
        for (unsigned i = 0; i < dy.size(); ++i) dy[i] = y[i] - target[i];
        Matrix dw = chip.check(gemm(x, dy, true, false, 128), {}, "training dW");
        chip.check(gemm(dy, w, false, true, 192), {}, "training dX");
        for (unsigned i = 0; i < w.size(); ++i) w[i] -= 0.5f * dw[i];
    }
    require(final_loss < initial * 0.001, "training must reduce loss by over 99.9 percent");
    std::cout << "PASS: 16x16 dense layer forward/dW/dX on chip, FP32 host SGD; epochs=" << epochs
              << " initial_loss=" << initial << " final_loss=" << final_loss << '\n';
}

static void benchmarks(Chip &chip) {
    std::mt19937 rng(8664);
    // One 32x32x32 GEMM: host fills the whole scratchpad, then drains every output tile.
    Gemm big = random_gemm(rng, 4, 4, 4, false, false, false);
    big.a_base = 0;
    big.b_base = 128;
    uint64_t begin = chip.cycles, begin_bytes = chip.written_bytes + chip.read_bytes;
    chip.check(big, {}, "GEMM32 benchmark");
    uint64_t cycles = chip.cycles - begin, bytes = chip.written_bytes + chip.read_bytes - begin_bytes;
    unsigned flops = 2 * 32 * 32 * 32;
    require(bytes == 2 * 32 * 32 * 2 + 32 * 32 * 4, "GEMM32 traffic accounting");
#ifdef TILE_V6
    // After the 1024-clock fill: 64 back-to-back 16-clock jobs plus fill, drain and handoff latency.
    require(cycles - 1024 >= 64 * 16 && cycles - 1024 <= 1150, "GEMM32 job throughput");
#endif
    std::cout << "BENCH design=" << design_name << " mode=gemm32_scratchpad cycles=" << cycles << " bytes=" << bytes
              << " flops=" << flops << " target_mflops=" << double(flops) / cycles * target_mhz << '\n';

    // Eight 16x16x16 GEMMs: the next operands load into the other scratchpad half while one runs.
    constexpr unsigned count = 8;
    std::vector<Gemm> gemms;
    for (unsigned n = 0; n < count; ++n) {
        Gemm g = random_gemm(rng, 2, 2, 2, false, false, false);
        g.a_base = (n & 1) * 128;
        g.b_base = g.a_base + 32;
        gemms.push_back(g);
    }
    begin = chip.cycles;
    begin_bytes = chip.written_bytes + chip.read_bytes;
    chip.store(gemms[0]);
    for (unsigned n = 0; n < count; ++n) {
        if (n + 1 < count) chip.writes = Chip::operand_writes(gemms[n + 1]);
        compare(chip.run(gemms[n]), reference(gemms[n]), "GEMM16 pipelined benchmark");
        while (chip.drive_write()) chip.tick();
        chip.dut.sp_write = 0;
    }
    cycles = chip.cycles - begin;
    bytes = chip.written_bytes + chip.read_bytes - begin_bytes;
    flops = count * 2 * 16 * 16 * 16;
    require(bytes == count * (2 * 16 * 16 * 2 + 16 * 16 * 4), "GEMM16 traffic accounting");
    std::cout << "BENCH design=" << design_name << " mode=gemm16_pipelined matrices=" << count << " cycles=" << cycles
              << " bytes=" << bytes << " flops=" << flops
              << " target_mflops=" << double(flops) / cycles * target_mhz << '\n';
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    try {
        Chip chip;
        shape_tests(chip);
        flag_tests(chip);
        protocol_tests(chip);
        training_test(chip);
        benchmarks(chip);
        chip.dut.final();
        std::cout << "PASS: " << design_name << " gemms=" << chip.gemms << " jobs=" << chip.jobs << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
