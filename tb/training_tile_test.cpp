#include "Vtraining_tile.h"
#include "verilated.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

using Matrix = std::array<float, 16>;

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

static void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

static Matrix reference(const Matrix &a, const Matrix &b, Matrix c,
                        bool ta, bool tb, bool accumulate) {
    if (!accumulate) c.fill(0.0f);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                c[i * 4 + j] = std::fma(a[ta ? k * 4 + i : i * 4 + k],
                                       b[tb ? j * 4 + k : k * 4 + j], c[i * 4 + j]);
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
    Vtraining_tile dut;
    unsigned jobs = 0;
    unsigned flags = 0;

    Tile() { reset(); }

    void tick() {
        dut.clk = 0;
        dut.eval();
        dut.clk = 1;
        dut.eval();
        dut.clk = 0;
        dut.eval();
    }

    void reset() {
        dut.rst = 1;
        dut.start = 0;
        dut.write_enable = 0;
        tick();
        require(!dut.busy && !dut.done && dut.exception_flags == 0, "reset control");
        dut.rst = 0;
    }

    void load(const Matrix &a, const Matrix &b, const Matrix &c) {
        require(!dut.busy, "load while busy");
        std::array<Matrix, 3> matrices{a, b, c};
        for (unsigned bank = 0; bank < 3; ++bank) {
            dut.write_enable = 1u << bank;
            for (unsigned i = 0; i < 16; ++i) {
                dut.write_addr = i;
                dut.write_data = bits(matrices[bank][i]);
                tick();
            }
        }
        dut.write_enable = 0;
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

    Matrix read() {
        Matrix out;
        for (unsigned i = 0; i < 16; ++i) {
            dut.read_addr = i;
            dut.eval();
            out[i] = value(dut.read_data);
        }
        return out;
    }

    Matrix finish(bool hostile = false) {
        for (unsigned cycle = 0; cycle < 64; ++cycle) {
            dut.start = hostile && cycle < 8;
            dut.write_enable = hostile && cycle < 8 ? 7 : 0;
            dut.write_addr = cycle % 16;
            dut.write_data = 0x7fc00000;
            tick();
            require(bool(dut.done) == (cycle == 63), "completion cycle");
            require(bool(dut.busy) == (cycle != 63), "busy duration");
        }
        flags = dut.exception_flags;
        Matrix out = read();
        tick();
        require(!dut.done && !dut.busy, "done must pulse once");
        compare(read(), out, "idle result hold");
        ++jobs;
        return out;
    }

    Matrix run(const Matrix &a, const Matrix &b, const Matrix &c = {},
               bool ta = false, bool tb = false, bool accumulate = false) {
        load(a, b, c);
        launch(ta, tb, accumulate);
        dut.transpose_a = !ta;
        dut.transpose_b = !tb;
        dut.accumulate = !accumulate;
        Matrix out = finish(true);
        compare(out, reference(a, b, c, ta, tb, accumulate), "matrix reference");
        return out;
    }
};

static void arithmetic_tests(Tile &tile) {
    struct Case { uint32_t a, b, c; unsigned flags; };
    const Case cases[] = {
        {0x3f800000, 0x40000000, 0, 0},
        {0x80000000, 0x3f800000, 0x80000000, 0},
        {0x00000001, 0x3f800000, 0, 0},
        {0x00000001, 0x3f000000, 0, 3},
        {0x7f7fffff, 0x40000000, 0, 5},
        {0x7f800000, 0x3f800000, 0, 0},
        {0x7f800000, 0, 0, 16},
        {0x7fc12345, 0x3f800000, 0, 0},
        {0x7f812345, 0x3f800000, 0, 16},
        {0x3f800000, 0x33800000, 0x3f800000, 1},
    };
    for (const auto &test : cases) {
        Matrix a, b, c;
        a.fill(value(test.a)); b.fill(value(test.b)); c.fill(value(test.c));
        tile.run(a, b, c, false, false, true);
        require(tile.flags == test.flags, "directed exception flags");
    }

    Matrix a{}, b{}, c;
    c.fill(-1.0f);
    for (int i = 0; i < 4; ++i) a[i * 4] = value(0x3f800001);
    for (int j = 0; j < 4; ++j) b[j] = value(0x3f7ffffe);
    auto fused = tile.run(a, b, c, false, false, true);
    require(bits(fused[0]) == 0xa8800000, "single rounding cancellation must retain -2^-46");

    std::mt19937 rng(20260912);
    for (unsigned test = 0; test < 256; ++test) {
        for (auto *matrix : {&a, &b, &c})
            for (float &x : *matrix)
                x = test < 128 ? float(int(rng() % 2001) - 1000) / 128.0f
                               : value(rng());
        tile.run(a, b, c, test & 1, test & 2, test & 4);
    }
    std::cout << "PASS: 267 numerical tiles; FP32 bit comparison, all transpose/accumulate modes, "
                 "NaNs, infinities, subnormals, ties, fused cancellation, exception flags\n";
}

static void protocol_tests(Tile &tile) {
    Matrix a{}, b{}, c{};
    for (int i = 0; i < 16; ++i) {
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

    for (unsigned abort_cycle : {0u, 1u, 31u, 63u}) {
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
    std::cout << "PASS: 64 compute cycles, busy command rejection, latched modes, start/reset priority, "
                 "abort/restart, retained operands, accumulated C, stable output\n";
}

static double loss(const Matrix &prediction, const Matrix &target) {
    double sum = 0;
    for (int i = 0; i < 16; ++i) {
        double diff = double(prediction[i]) - target[i];
        sum += 0.5 * diff * diff;
    }
    return sum;
}

static void training_test(Tile &tile) {
    Matrix x{1, 0.25f, 0, 0, 0, 1, -0.25f, 0,
             0, 0, 1, 0.25f, -0.25f, 0, 0, 1};
    Matrix target_weights, weights{};
    for (int i = 0; i < 16; ++i) target_weights[i] = float(i - 7) / 8;
    Matrix target = reference(x, target_weights, {}, false, false, false);
    Matrix step{};
    for (int i = 0; i < 4; ++i) step[5 * i] = -0.125f;
    double initial = 0, previous = 0;
    for (int epoch = 0; epoch < 33; ++epoch) {
        Matrix y = tile.run(x, weights);
        double current = loss(y, target);
        if (epoch == 0) initial = current;
        else require(current < previous, "training loss must decrease");
        previous = current;
        if (epoch == 32) break;
        Matrix dy;
        for (int i = 0; i < 16; ++i) dy[i] = y[i] - target[i];
        Matrix dw = tile.run(x, dy, {}, true, false);
        Matrix dx = tile.run(dy, weights, {}, false, true);
        // Check the gradient formula independently of the FMA reference loop.
        if (epoch == 1) {
            constexpr float epsilon = 1.0f / 1024;
            for (int i = 0; i < 16; ++i) {
                Matrix plus = weights, minus = weights;
                plus[i] += epsilon; minus[i] -= epsilon;
                double numerical = (loss(reference(x, plus, {}, false, false, false), target) -
                                    loss(reference(x, minus, {}, false, false, false), target)) / (2 * epsilon);
                require(std::abs(numerical - dw[i]) < 0.003, "weight gradient finite difference");
                plus = x; minus = x;
                plus[i] += epsilon; minus[i] -= epsilon;
                numerical = (loss(reference(plus, weights, {}, false, false, false), target) -
                             loss(reference(minus, weights, {}, false, false, false), target)) / (2 * epsilon);
                require(std::abs(numerical - dx[i]) < 0.003, "input gradient finite difference");
            }
        }
        weights = tile.run(step, dw, weights, false, false, true);
    }
    require(previous < initial * 0.001, "training must reduce loss by over 99.9 percent");
    std::cout << "PASS: dense-layer forward, dW, dX, SGD; finite-difference gradients; epochs=32"
              << " initial_loss=" << initial << " final_loss=" << previous << '\n';
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    try {
        Tile tile;
        arithmetic_tests(tile);
        protocol_tests(tile);
        training_test(tile);
        tile.dut.final();
        std::cout << "PASS: training_tile jobs=" << tile.jobs << '\n';
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
