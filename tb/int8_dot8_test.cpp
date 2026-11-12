#include "Vint8_dot8_int32.h"
#include "verilated.h"
#include <cstdint>
#include <deque>
#include <iostream>
#include <random>

// Random and extreme INT8 operands against a wrapping INT32 reference; two-clock latency.
int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    Vint8_dot8_int32 dut;
    std::mt19937 rng(8);
    std::deque<uint32_t> expected;
    constexpr unsigned vectors = 1000000, latency = 2;
    for (unsigned n = 0; n < vectors + latency; ++n) {
        if (n < vectors) {
            uint64_t a = 0, b = 0;
            uint32_t sum = n % 4 == 0 ? 0x7fffff00u + (rng() & 0xff) : rng();
            uint32_t c = sum;
            for (unsigned k = 0; k < 8; ++k) {
                int8_t x = n % 8 == 1 ? int8_t(-128) : int8_t(rng()),
                       y = n % 8 == 1 ? int8_t(-128) : int8_t(rng());
                a |= uint64_t(uint8_t(x)) << (8 * k);
                b |= uint64_t(uint8_t(y)) << (8 * k);
                sum += uint32_t(int32_t(x) * int32_t(y));
            }
            dut.a = a;
            dut.b = b;
            dut.c = c;
            expected.push_back(sum);
        }
        dut.clk = 0;
        dut.eval();
        dut.clk = 1;
        dut.eval();
        if (n + 1 >= latency && !expected.empty()) {
            if (dut.result != expected.front()) {
                std::cerr << "FAIL: vector " << n - latency << " got " << dut.result << " expected "
                          << expected.front() << '\n';
                return 1;
            }
            expected.pop_front();
        }
    }
    std::cout << "PASS: int8_dot8_int32 vectors=" << vectors
              << " including -128*-128 and INT32 wraparound\n";
    return 0;
}
