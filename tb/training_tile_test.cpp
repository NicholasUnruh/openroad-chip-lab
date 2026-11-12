#include "Vtraining_tile_v4.h"
using Model = Vtraining_tile_v4;
constexpr unsigned compute_cycles = 67;
constexpr double target_mhz = 40;
constexpr const char *design_name = "training_tile_v4";
#include "vcd_window.h"
#include "verilated.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

constexpr unsigned dimension = 8;
constexpr unsigned elements = dimension * dimension;
constexpr unsigned tile_flops = 2 * dimension * dimension * dimension;
using Matrix = std::array<float, elements>;

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

static Matrix bf16_matrix(Matrix matrix) {
    for (auto &x : matrix) x = bf16(x);
    return matrix;
}

static void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

static Matrix reference(const Matrix &a, const Matrix &b, Matrix c, bool ta, bool tb,
                        bool accumulate) {
    if (!accumulate) c.fill(0.0f);
    for (int i = 0; i < dimension; ++i)
        for (int j = 0; j < dimension; ++j)
            for (int k = 0; k < dimension; ++k)
                c[i * dimension + j] =
                    std::fma(a[ta ? k * dimension + i : i * dimension + k],
                             b[tb ? j * dimension + k : k * dimension + j], c[i * dimension + j]);
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

class Tile {
   public:
    Model dut;
    unsigned jobs = 0;
    unsigned flags = 0;
    uint64_t cycles = 0, written_bytes = 0, read_bytes = 0;
    VcdWindow vcd;

    Tile() { reset(); }

    void tick() {
        dut.clk = 0;
        dut.eval();
        vcd.edge(cycles, false);
        dut.clk = 1;
        dut.eval();
        vcd.edge(cycles, true);
        ++cycles;
        dut.clk = 0;
        dut.eval();
    }

    void reset() {
        dut.job_bank = 0;
        dut.host_bank = 0;
        dut.rst = 1;
        dut.start = 0;
        dut.write_enable = 0;
        tick();
        require(!dut.busy && !dut.done && dut.exception_flags == 0, "reset control");
        dut.rst = 0;
    }

    void load_bank(unsigned bank, const Matrix &matrix) {
        require(!dut.busy, "load while busy");
        dut.write_enable = 1u << bank;
        for (unsigned i = 0; i < elements; ++i) {
            dut.write_addr = i;
            if (bank < 2) {
                dut.write_data =
                    (bits(bf16(matrix[i])) >> 16) | (bits(bf16(matrix[i + 1])) & 0xffff0000u);
                ++i;
            } else
                dut.write_data = bits(matrix[i]);
            dut.eval();
            require(dut.write_ready, "idle write ready");
            tick();
            written_bytes += 4;
        }
        dut.write_enable = 0;
    }

    void load(const Matrix &a, const Matrix &b, const Matrix &c) {
        load_bank(0, a);
        load_bank(1, b);
        load_bank(2, c);
    }

    void launch(bool ta, bool tb, bool accumulate) {
        dut.transpose_a = ta;
        dut.transpose_b = tb;
        dut.accumulate = accumulate;
        dut.start = 1;
        tick();
        dut.start = 0;
        require(dut.busy && !dut.done && dut.exception_flags == 0, "launch control");
    }

    Matrix read(bool clocked = false) {
        Matrix out;
        for (unsigned i = 0; i < elements; ++i) {
            dut.read_addr = i;
            if (clocked) {
                tick();
                read_bytes += 4;
            } else
                dut.eval();
            out[i] = value(dut.read_data);
        }
        return out;
    }

    Matrix finish(bool hostile = false) {
        for (unsigned cycle = 0; cycle < compute_cycles; ++cycle) {
            bool poison = hostile && (cycle < 8 || cycle >= compute_cycles - 3);
            dut.start = poison;
            dut.write_enable = poison ? 7 : 0;
            dut.write_addr = cycle % elements;
            dut.write_data = 0x7fc00000;
            tick();
            require(bool(dut.done) == (cycle == compute_cycles - 1), "completion cycle");
            require(bool(dut.busy) == (cycle != compute_cycles - 1), "busy duration");
        }
        dut.start = 0;
        dut.write_enable = 0;
        flags = dut.exception_flags;
        Matrix out = read();
        tick();
        require(!dut.done && !dut.busy, "done must pulse once");
        compare(read(), out, "idle result hold");
        ++jobs;
        return out;
    }

    Matrix run(const Matrix &a, const Matrix &b, const Matrix &c = {}, bool ta = false,
               bool tb = false, bool accumulate = false) {
        load(a, b, c);
        launch(ta, tb, accumulate);
        dut.transpose_a = !ta;
        dut.transpose_b = !tb;
        dut.accumulate = !accumulate;
        Matrix out = finish(true);
        compare(out, reference(bf16_matrix(a), bf16_matrix(b), c, ta, tb, accumulate),
                "BF16 matrix reference");
        return out;
    }
};

static void protocol_tests(Tile &tile) {
    Matrix a{}, b{}, c{};
    for (int i = 0; i < elements; ++i) {
        a[i] = float(i - 8) / 4;
        b[i] = float(i + 1) / 8;
        c[i] = 2;
    }
    tile.load(a, b, c);
    // Start wins over a simultaneous attempted write to all three banks.
    tile.dut.write_enable = 7;
    tile.dut.write_addr = 0;
    tile.dut.write_data = 0x7fc00000;
    tile.launch(false, false, true);
    tile.dut.write_enable = 0;
    auto first = tile.finish();
    compare(first, reference(a, b, c, false, false, true), "start priority");
    tile.launch(true, true, true);
    compare(tile.finish(), reference(a, b, first, true, true, true), "reuse accumulated C");

    for (unsigned abort_cycle : {0u, 1u, 2u, 3u, compute_cycles / 2, compute_cycles - 1}) {
        tile.launch(false, false, false);
        for (unsigned i = 0; i < abort_cycle; ++i) tile.tick();
        tile.dut.start = 1;
        tile.dut.write_enable = 7;
        tile.dut.rst = 1;
        tile.tick();
        require(!tile.dut.busy && !tile.dut.done, "reset must abort work");
        tile.reset();
        tile.launch(false, false, false);
        compare(tile.finish(), reference(a, b, {}, false, false, false), "restart after abort");
    }
    std::cout << "PASS: " << compute_cycles
              << " compute cycles, busy command rejection, latched modes, start/reset priority, "
                 "abort/restart, retained operands, accumulated C, stable output\n";
}

static double loss(const Matrix &prediction, const Matrix &target) {
    double sum = 0;
    for (int i = 0; i < elements; ++i) {
        double diff = double(prediction[i]) - target[i];
        sum += 0.5 * diff * diff;
    }
    return sum;
}

#include "bf16_tile_tests.h"

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    try {
        if (VcdWindow::requested()) Verilated::traceEverOn(true);
        Tile tile;
        if (!VcdWindow::requested()) {
            bf16_tests(tile);
            protocol_tests(tile);
            buffered_tests(tile);
            bf16_training_test(tile);
            larger_matrix_test(tile);
            streaming_benchmark(tile);
            streamed_gemm_benchmark<64, 8>(tile);
            streamed_gemm_benchmark<8, 16>(tile);
        }
        streamed_gemm_benchmark<1, 32>(tile);
        tile.dut.final();
        std::cout << "PASS: " << design_name << " jobs=" << tile.jobs << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
