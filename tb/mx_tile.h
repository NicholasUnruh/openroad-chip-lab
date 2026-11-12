// Host driver and bit-exact reference for training_tile_v7 (docs/training-v7.md), shared by the
// GEMM regression (tb/mx_tile_test.cpp) and the training run (tb/mx_tile_train.cpp).
#pragma once
#include "Vtraining_tile_v7.h"
#include "mx_model.h"
#include "verilated.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

inline void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

inline float fp32_value(uint32_t raw) {
    float result;
    std::memcpy(&result, &raw, sizeof(result));
    return result;
}

// C[8m x 8n] = A B^T over K = 32k: A is 8m x K and B is 8n x K, both K-contiguous E4M3 bytes, with
// one E8M0 scale per row and K block (sa[row * k + block]).
struct MxGemm {
    unsigned m, n, k;
    unsigned a_base = 0, b_base = 0, a_scale_base = 0, b_scale_base = 0;
    std::vector<uint8_t> a, b, sa, sb;
    unsigned inner() const { return 32 * k; }
    unsigned a_sp_rows() const { return 16 * m * k; }
    unsigned b_sp_rows() const { return 16 * n * k; }
    unsigned scale_sp_rows(unsigned tiles) const { return tiles * k; }
    unsigned sp_rows() const { return a_sp_rows() + b_sp_rows() + (m + n) * k; }
    // Places A, A scales, B, B scales one after another from row `base`.
    void pack(unsigned base) {
        a_base = base % 256;
        a_scale_base = (a_base + a_sp_rows()) % 256;
        b_base = (a_scale_base + scale_sp_rows(m)) % 256;
        b_scale_base = (b_base + b_sp_rows()) % 256;
    }
};

// Each output chains k block dot products from +0, rounding once per block.
inline std::vector<uint32_t> reference(const MxGemm &g) {
    unsigned rows = 8 * g.m, cols = 8 * g.n;
    std::vector<uint32_t> c(rows * cols);
    std::array<uint8_t, 32> x, y;
    for (unsigned i = 0; i < rows; ++i)
        for (unsigned j = 0; j < cols; ++j) {
            uint32_t acc = 0;
            for (unsigned kb = 0; kb < g.k; ++kb) {
                std::memcpy(x.data(), &g.a[i * g.inner() + 32 * kb], 32);
                std::memcpy(y.data(), &g.b[j * g.inner() + 32 * kb], 32);
                acc = mx_dot32(x, y, g.sa[i * g.k + kb], g.sb[j * g.k + kb], acc, MxElement::e4m3)
                          .bits;
            }
            c[i * cols + j] = acc;
        }
    return c;
}

inline void compare(const std::vector<uint32_t> &got, const std::vector<uint32_t> &expected,
                    const std::string &label) {
    require(got.size() == expected.size(), label + " size");
    for (size_t i = 0; i < got.size(); ++i)
        require(got[i] == expected[i], label + " element=" + std::to_string(i) +
                                           " got_bits=" + std::to_string(got[i]) +
                                           " expected_bits=" + std::to_string(expected[i]));
}

// Host behaviour while draining: probability of skipping a read cycle and maximum ack delay.
struct HostPolicy {
    unsigned stall_percent = 0, max_ack_delay = 0;
};

using Writes = std::deque<std::pair<unsigned, uint32_t>>;

class Chip {
   public:
    Vtraining_tile_v7 dut;
    std::mt19937 rng{20261111};
    uint64_t cycles = 0, written_bytes = 0, read_bytes = 0;
    unsigned gemms = 0, jobs = 0;
    Writes writes;  // overlapped scratchpad writes

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

    // Bytes packed little-endian into words from scratchpad row `row`.
    static void queue_bytes(Writes &out, const uint8_t *data, unsigned count, unsigned row) {
        for (unsigned i = 0; i < count; i += 4) {
            uint32_t word = 0;
            for (unsigned b = 0; b < 4 && i + b < count; ++b)
                word |= uint32_t(data[i + b]) << 8 * b;
            out.emplace_back((4 * row + i / 4) & 0x3ffu, word);
        }
    }

    // One scale row per (tile, K block): bytes 0-7 hold the tile's eight row scales.
    static void queue_scales(Writes &out, const std::vector<uint8_t> &scales, unsigned tiles,
                             unsigned blocks, unsigned base) {
        for (unsigned t = 0; t < tiles; ++t)
            for (unsigned kb = 0; kb < blocks; ++kb) {
                uint8_t row[8];
                for (unsigned r = 0; r < 8; ++r) row[r] = scales[(8 * t + r) * blocks + kb];
                queue_bytes(out, row, 8, base + t * blocks + kb);
            }
    }

    static Writes operand_writes(const MxGemm &g) {
        Writes out;
        queue_bytes(out, g.a.data(), unsigned(g.a.size()), g.a_base);
        queue_bytes(out, g.b.data(), unsigned(g.b.size()), g.b_base);
        queue_scales(out, g.sa, g.m, g.k, g.a_scale_base);
        queue_scales(out, g.sb, g.n, g.k, g.b_scale_base);
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

    void store(const MxGemm &g) {
        writes = operand_writes(g);
        for (unsigned limit = 0; drive_write(); ++limit) {
            require(limit < 1024, "store bound");
            tick();
        }
        dut.sp_write = 0;
    }

    void command(const MxGemm &g) {
        dut.m_tiles = g.m - 1;
        dut.n_tiles = g.n - 1;
        dut.k_tiles = g.k - 1;
        dut.a_base = g.a_base;
        dut.b_base = g.b_base;
        dut.a_scale_base = g.a_scale_base;
        dut.b_scale_base = g.b_scale_base;
    }

    // Runs one accepted GEMM to completion, draining tiles under the host policy.
    // hijack_cycle issues a conflicting start while busy; abort_cycle resets mid-run.
    std::vector<uint32_t> run(const MxGemm &g, HostPolicy policy = {}, int hijack_cycle = -1,
                              int abort_cycle = -1) {
        command(g);
        dut.gemm_start = 1;
        tick();
        dut.gemm_start = 0;
        require(dut.gemm_busy && !dut.gemm_done, "GEMM accept");
        std::vector<uint32_t> out(64 * g.m * g.n);
        unsigned tiles = g.m * g.n, next_tile = 0, count = 0, ack_wait = 0;
        int current = -1;
        for (unsigned cycle = 0;; ++cycle) {
            require(cycle < 200000, "GEMM timeout");
            drive_write();
            if (int(cycle) == hijack_cycle) {
                MxGemm other = g;
                other.m = 1 + (g.m % 4);
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
                    unsigned row = 8 * (current >> 2) + count / 8,
                             col = 8 * (current & 3) + count % 8;
                    out[row * 8 * g.n + col] = dut.read_data;
                    ++count;
                    read_bytes += 4;
                }
                if (count == 64) {
                    if (ack_wait)
                        --ack_wait;
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

    std::vector<uint32_t> check(const MxGemm &g, HostPolicy policy = {},
                                const std::string &label = "GEMM") {
        store(g);
        std::vector<uint32_t> out = run(g, policy);
        compare(out, reference(g), label);
        return out;
    }
};
