#include <algorithm>
#include <vector>

static void bf16_tests(Tile &tile) {
    require(bits(bf16(value(0x3f808000))) == 0x3f800000, "BF16 even tie");
    require(bits(bf16(value(0x3f818000))) == 0x3f820000, "BF16 odd tie");
    struct Case { uint16_t a, b; uint32_t c; unsigned flags; };
    const Case cases[] = {
        {0x3f80, 0x4000, 0, 0}, {0x8000, 0x3f80, 0x80000000, 0},
        {0x0001, 0x3f80, 0, 0}, {0x0001, 0x0001, 0, 3},
        {0x7f7f, 0x4000, 0, 5}, {0x7f80, 0x3f80, 0, 0},
        {0x7f80, 0x0000, 0, 16}, {0x7fc1, 0x3f80, 0, 0},
        {0x7f81, 0x3f80, 0, 16}, {0x3f80, 0x3380, 0x3f800000, 1}
    };
    Matrix a, b, c;
    for (const auto &test : cases) {
        a.fill(value(uint32_t(test.a) << 16));
        b.fill(value(uint32_t(test.b) << 16));
        c.fill(value(test.c));
        tile.run(a, b, c, false, false, true);
        require(tile.flags == test.flags, "BF16 directed exception flags");
    }
    // These small product terms must survive in C below BF16's rounding precision.
    a.fill(0); b.fill(0); c.fill(1);
    a[0] = 1; b[0] = value(0x3b800000);
    auto out = tile.run(a, b, c, false, false, true);
    require(bits(out[0]) == 0x3f808000, "FP32 accumulation retains half a BF16 ULP");
    for (unsigned lane = 0; lane < dimension; ++lane) {
        a.fill(0); b.fill(0); c.fill(0);
        b[lane] = value(0x7f810000);
        tile.run(a, b, c);
        require(tile.flags == 16, "invalid exception from every lane");
    }
    a[0] = value(0x7f7f0000);
    b[0] = 2;
    tile.run(a, b, c);
    require(tile.flags == 21, "combine invalid and overflow across first/last lanes");
    std::mt19937 rng(20260913);
    for (unsigned test = 0; test < 512; ++test) {
        for (int i = 0; i < elements; ++i) {
            a[i] = test < 256 ? bf16(float(int(rng() % 2001) - 1000) / 128) : value(rng() << 16);
            b[i] = test < 256 ? bf16(float(int(rng() % 2001) - 1000) / 128) : value(rng() << 16);
            c[i] = test < 256 ? float(int(rng() % 2001) - 1000) / 127 : value(rng());
        }
        tile.dut.host_bank = tile.dut.job_bank = test & 1;
        tile.run(a, b, c, test & 1, test & 2, test & 4);
    }
    tile.dut.host_bank = tile.dut.job_bank = 0;
    std::cout << "PASS: " << 524 + dimension << " native BF16 numerical tiles; FP32 FMA reference, all modes, "
                 "special values, subnormals, exception flags, FP32 accumulation precision\n";
    for (unsigned base = 0; base < 65536; base += elements) {
        constexpr float factors[] = {1.0f, -0.5f, 1.0f / 1024, 1.9921875f};
        for (unsigned i = 0; i < elements; ++i) {
            a[i] = value((base + i) << 16);
            b[i] = factors[i % 4];
            c[i] = float(int(i) - 8) / 127;
        }
        tile.dut.host_bank = tile.dut.job_bank = (base / elements) & 1;
        tile.run(a, b, c, base & elements, base & (2*elements), base & (4*elements));
    }
    tile.dut.host_bank = tile.dut.job_bank = 0;
    std::cout << "PASS: all 65536 BF16 A encodings in " << 65536 / elements << " tiles against FP32 FMA reference\n";
}

static void host_write(Tile &tile, unsigned bank, unsigned mask, unsigned addr,
                       uint32_t data, bool accepted) {
    tile.dut.host_bank = bank;
    tile.dut.write_enable = mask;
    tile.dut.write_addr = addr;
    tile.dut.write_data = data;
    tile.dut.eval();
    require(bool(tile.dut.write_ready) == accepted, "host write ownership/alignment");
    tile.tick();
    tile.dut.write_enable = 0;
}

static void buffered_tests(Tile &tile) {
    Matrix a, b, c, other;
    for (unsigned i = 0; i < elements; ++i) {
        a[i] = float(int(i) - 8) / 4; b[i] = float(i + 1) / 8;
        c[i] = 0.25f; other[i] = -float(i + 1);
    }
    tile.dut.host_bank = 0;
    tile.load(a, b, c);
    // Reject a whole malformed command: odd BF16 address or mixed operand/C mask.
    host_write(tile, 0, 1, 1, 0x7fc07fc0, false);
    host_write(tile, 0, 5, 0, 0x7fc07fc0, false);
    tile.dut.job_bank = 0;
    tile.launch(false, false, true);
    compare(tile.finish(), reference(a, b, c, false, false, true), "invalid writes preserve banks");

    for (unsigned active = 0; active < 2; ++active) {
        tile.dut.host_bank = active;
        tile.load(a, b, c);
        tile.dut.host_bank = !active;
        tile.load(b, a, other);
        tile.dut.job_bank = active;
        tile.dut.write_enable = 3;
        tile.dut.write_addr = 0;
        tile.dut.write_data = 0x7fc07fc0;
        tile.dut.start = 1;
        tile.dut.eval();
        require(!tile.dut.write_ready, "start takes priority over inactive writes");
        tile.launch(false, true, true);
        tile.dut.write_enable = 0;
        tile.dut.job_bank = !active;
        tile.dut.transpose_a = 1;
        tile.dut.transpose_b = 0;
        for (unsigned cycle = 0; cycle < compute_cycles; ++cycle) {
            tile.dut.start = 1;
            if (cycle < elements)
                host_write(tile, !active, 4, cycle, bits(c[cycle]), true);
            else
                host_write(tile, active, 3, 0, 0x7fc07fc0, false);
            require(bool(tile.dut.done) == (cycle == compute_cycles - 1), "overlap completion");
        }
        tile.dut.start = 0;
        tile.dut.host_bank = active;
        compare(tile.read(), reference(a, b, c, false, true, true), "busy ownership and latched bank");
        tile.dut.host_bank = !active;
        compare(tile.read(), c, "inactive C writable throughout compute");
        tile.dut.job_bank = !active;
        tile.launch(true, false, true);
        compare(tile.finish(), reference(b, a, c, true, false, true), "immediate bank switch");
        ++tile.jobs;
    }

    // Abort at every pipeline phase; reset must suppress simultaneous inactive-bank writes.
    for (unsigned phase = 0; phase < compute_cycles; ++phase) {
        tile.dut.host_bank = 1;
        tile.load(b, a, other);
        tile.dut.host_bank = 0;
        tile.load(a, b, c);
        tile.dut.job_bank = 0;
        tile.launch(false, false, false);
        for (unsigned k = 0; k < phase; ++k) tile.tick();
        tile.dut.rst = 1;
        host_write(tile, 1, 1u << (phase % 3), 0, 0x7fc07fc0, false);
        require(!tile.dut.busy && !tile.dut.done && !tile.dut.exception_flags, "buffered abort");
        tile.dut.rst = 0;
        compare(tile.read(), other, "reset preserves inactive bank");
        tile.dut.job_bank = 1;
        tile.launch(false, false, true);
        compare(tile.finish(), reference(b, a, other, false, false, true), "restart on alternate bank");
    }
    tile.dut.host_bank = tile.dut.job_bank = 0;
    std::cout << "PASS: dual-bank ownership, concurrent C writes, rejected malformed writes, "
                 "latched bank, immediate switch, all " << compute_cycles << " abort phases\n";
}

static void bf16_training_test(Tile &tile) {
    Matrix x{};
    for (unsigned i = 0; i < dimension; ++i) {
        x[i * dimension + i] = 1;
        x[i * dimension + (i + 1) % dimension] = i % 2 ? -0.25f : 0.25f;
    }
    Matrix target_weights, weights{}, step{};
    for (unsigned i = 0; i < elements; ++i) target_weights[i] = float(int(i) - int(elements / 2 - 1)) / (2 * dimension);
    for (unsigned i = 0; i < dimension; ++i) step[i * (dimension + 1)] = -0.125f;
    Matrix target = reference(x, target_weights, {}, false, false, false);
    double initial = 0, final_loss = 0;
    for (unsigned epoch = 0; epoch <= 48; ++epoch) {
        Matrix y = tile.run(x, weights);
        final_loss = loss(y, target);
        if (!epoch) initial = final_loss;
        if (epoch == 48) break;
        Matrix dy;
        for (unsigned i = 0; i < elements; ++i) dy[i] = y[i] - target[i];
        Matrix dw = tile.run(x, dy, {}, true, false);
        tile.run(dy, weights, {}, false, true);
        weights = tile.run(step, dw, weights, false, false, true);
    }
    require(final_loss < initial * 0.001, "BF16 training reduces loss by over 99.9 percent");
    std::cout << "PASS: BF16 forward/dW/dX/SGD with FP32 master weights; epochs=48 initial_loss="
              << initial << " final_loss=" << final_loss << '\n';
}

static void larger_matrix_test(Tile &tile) {
    constexpr unsigned rows = 7, cols = 9, inner = 13;
    std::array<float, rows * inner> a;
    std::array<float, inner * cols> b;
    std::array<float, rows * cols> expected{}, actual{};
    std::mt19937 rng(1949);
    for (auto &x : a) x = bf16(float(int(rng() % 257) - 128) / 37);
    for (auto &x : b) x = bf16(float(int(rng() % 257) - 128) / 53);
    for (unsigned i = 0; i < rows; ++i)
        for (unsigned j = 0; j < cols; ++j)
            for (unsigned k = 0; k < inner; ++k)
                expected[i * cols + j] = std::fma(a[i * inner + k], b[k * cols + j],
                                                 expected[i * cols + j]);
    for (unsigned row = 0; row < rows; row += dimension) {
        for (unsigned col = 0; col < cols; col += dimension) {
            tile.dut.host_bank = tile.dut.job_bank = ((row / dimension) * ((cols + dimension - 1) / dimension) + col / dimension) & 1;
            Matrix result;
            for (unsigned k = 0; k < inner; k += dimension) {
                Matrix block_a{}, block_b{};
                for (unsigned i = 0; i < dimension; ++i)
                    for (unsigned j = 0; j < dimension; ++j) {
                        if (row + i < rows && k + j < inner)
                            block_a[i * dimension + j] = a[(row + i) * inner + k + j];
                        if (k + i < inner && col + j < cols)
                            block_b[i * dimension + j] = b[(k + i) * cols + col + j];
                    }
                tile.load_bank(0, block_a);
                tile.load_bank(1, block_b);
                tile.launch(false, false, k != 0);
                result = tile.finish();
            }
            for (unsigned i = 0; i < dimension && row + i < rows; ++i)
                for (unsigned j = 0; j < dimension && col + j < cols; ++j)
                    actual[(row + i) * cols + col + j] = result[i * dimension + j];
        }
    }
    for (unsigned i = 0; i < actual.size(); ++i)
        require(bits(actual[i]) == bits(expected[i]), "larger GEMM output");
    tile.dut.host_bank = tile.dut.job_bank = 0;
    std::cout << "PASS: host-tiled 7x13 times 13x9 GEMM; "
              << ((rows + dimension - 1) / dimension) * ((cols + dimension - 1) / dimension) *
                 ((inner + dimension - 1) / dimension)
              << " jobs, retained hardware C, padded boundaries\n";
}

static void streamed_gemm8_benchmark(Tile &tile) {
    constexpr unsigned count = 64, size = 8, blocks = size / dimension;
    static_assert(size % dimension == 0, "benchmark must tile exactly");
    struct Job { Matrix a{}, b{}, expected{}; bool accumulate, final; };
    std::vector<Job> jobs;
    std::mt19937 rng(8664);
    for (unsigned matrix = 0; matrix < count; ++matrix) {
        std::array<float, size * size> a, b, expected{};
        for (unsigned i = 0; i < size * size; ++i) {
            a[i] = bf16(float(int(rng() % 257) - 128) / 37);
            b[i] = bf16(float(int(rng() % 257) - 128) / 53);
        }
        for (unsigned i = 0; i < size; ++i)
            for (unsigned j = 0; j < size; ++j)
                for (unsigned k = 0; k < size; ++k)
                    expected[i * size + j] = std::fma(a[i * size + k], b[k * size + j],
                                                     expected[i * size + j]);
        // Alternate two output blocks across K chunks so each buffer retains its partial C.
        for (unsigned group = 0; group < blocks * blocks; group += 2)
            for (unsigned k = 0; k < blocks; ++k)
                for (unsigned slot = 0; slot < 2 && group + slot < blocks * blocks; ++slot) {
                    unsigned row = ((group + slot) / blocks) * dimension;
                    unsigned col = ((group + slot) % blocks) * dimension;
                    Job job;
                    job.accumulate = k != 0;
                    job.final = k == blocks - 1;
                    for (unsigned i = 0; i < dimension; ++i)
                        for (unsigned j = 0; j < dimension; ++j) {
                            job.a[i * dimension + j] = a[(row + i) * size + k * dimension + j];
                            job.b[i * dimension + j] = b[(k * dimension + i) * size + col + j];
                            job.expected[i * dimension + j] = expected[(row + i) * size + col + j];
                        }
                    jobs.push_back(job);
                }
    }
    tile.reset();
    uint64_t begin = tile.cycles, begin_bytes = tile.written_bytes + tile.read_bytes;
    tile.load_bank(0, jobs[0].a); tile.load_bank(1, jobs[0].b);
    for (unsigned n = 0; n < jobs.size(); ++n) {
        tile.dut.job_bank = n & 1;
        tile.launch(false, false, jobs[n].accumulate);
        tile.dut.host_bank = !(n & 1);
        Matrix previous;
        for (unsigned cycle = 0; cycle < compute_cycles; ++cycle) {
            if (n && jobs[n - 1].final && cycle < elements) {
                tile.dut.read_addr = cycle;
                tile.dut.eval();
                previous[cycle] = value(tile.dut.read_data);
                tile.read_bytes += 4;
            }
            if (n + 1 < jobs.size() && cycle < elements) {
                unsigned bank = cycle < elements / 2 ? 0 : 1;
                unsigned addr = (cycle % (elements / 2)) * 2;
                const auto &input = bank == 0 ? jobs[n + 1].a : jobs[n + 1].b;
                uint32_t data = (bits(input[addr]) >> 16) | (bits(input[addr + 1]) & 0xffff0000u);
                host_write(tile, !(n & 1), 1u << bank, addr, data, true);
                tile.written_bytes += 4;
            } else tile.tick();
            require(bool(tile.dut.done) == (cycle == compute_cycles - 1), "GEMM8 completion");
            require(bool(tile.dut.busy) == (cycle < compute_cycles - 1), "GEMM8 busy");
        }
        if (n && jobs[n - 1].final) compare(previous, jobs[n - 1].expected, "GEMM8 streamed reference");
    }
    tile.dut.host_bank = (jobs.size() - 1) & 1;
    compare(tile.read(true), jobs.back().expected, "GEMM8 final drain");
    uint64_t cycles = tile.cycles - begin;
    uint64_t bytes = tile.written_bytes + tile.read_bytes - begin_bytes;
    constexpr unsigned flops = count * 2 * size * size * size;
    require(cycles == 2 * elements + jobs.size() * (compute_cycles + 1), "GEMM8 cycle accounting");
    require(bytes == jobs.size() * elements * 4 + count * size * size * 4, "GEMM8 traffic accounting");
    std::cout << "BENCH design=" << design_name << " mode=gemm8_stream matrices=" << count
              << " jobs=" << jobs.size() << " cycles=" << cycles << " bytes=" << bytes
              << " flops=" << flops << " target_mflops=" << double(flops) / cycles * target_mhz << '\n';
    tile.dut.host_bank = tile.dut.job_bank = 0;
}

static void streaming_benchmark(Tile &tile) {
    constexpr unsigned count = 64;
    std::array<Matrix, count> a, b, c, expected;
    std::mt19937 rng(713);
    for (unsigned n = 0; n < count; ++n)
        for (unsigned i = 0; i < elements; ++i) {
            a[n][i] = bf16(float(int(rng() % 257) - 128) / 32);
            b[n][i] = bf16(float(int(rng() % 257) - 128) / 64);
            c[n][i] = float(int(rng() % 257) - 128) / 127;
        }
    for (bool acc : {false, true}) {
        for (unsigned n = 0; n < count; ++n)
            expected[n] = reference(a[n], b[n], c[n], n & 1, n & 2, acc);
        for (bool overlap : {false, true}) {
            tile.reset();
            uint64_t begin = tile.cycles;
            uint64_t begin_bytes = tile.written_bytes + tile.read_bytes;
            unsigned loads = acc ? 2 * elements : elements;
            if (!overlap) {
                for (unsigned n = 0; n < count; ++n) {
                    tile.load_bank(0, a[n]); tile.load_bank(1, b[n]);
                    if (acc) tile.load_bank(2, c[n]);
                    tile.launch(n & 1, n & 2, acc);
                    for (unsigned k = 0; k < compute_cycles; ++k) tile.tick();
                    require(tile.dut.done && !tile.dut.busy, "serial benchmark completion");
                    compare(tile.read(true), expected[n], "serial benchmark");
                }
            } else {
                tile.load_bank(0, a[0]); tile.load_bank(1, b[0]);
                if (acc) tile.load_bank(2, c[0]);
                for (unsigned n = 0; n < count; ++n) {
                    tile.dut.job_bank = n & 1;
                    tile.launch(n & 1, n & 2, acc);
                    tile.dut.host_bank = !(n & 1);
                    Matrix previous;
                    unsigned duration = n + 1 < count ? std::max(loads, compute_cycles) : compute_cycles;
                    for (unsigned k = 0; k < duration; ++k) {
                        if (n && k < elements) {
                            tile.dut.read_addr = k;
                            tile.dut.eval();
                            previous[k] = value(tile.dut.read_data);
                            tile.read_bytes += 4;
                        }
                        if (n + 1 < count && k < loads) {
                            unsigned bank = k < elements / 2 ? 0 : k < elements ? 1 : 2;
                            unsigned addr = bank < 2 ? (k % (elements / 2)) * 2 : k - elements;
                            const Matrix &input = bank == 0 ? a[n + 1] : bank == 1 ? b[n + 1] : c[n + 1];
                            uint32_t data = bank < 2 ? (bits(input[addr]) >> 16) |
                                                       (bits(input[addr + 1]) & 0xffff0000u)
                                                    : bits(input[addr]);
                            host_write(tile, !(n & 1), 1u << bank, addr, data, true);
                            tile.written_bytes += 4;
                        } else tile.tick();
                        require(bool(tile.dut.done) == (k == compute_cycles - 1), "streamed done pulse");
                        require(bool(tile.dut.busy) == (k < compute_cycles - 1), "streamed busy duration");
                    }
                    if (n) compare(previous, expected[n - 1], "concurrent result read");
                }
                tile.dut.host_bank = (count - 1) & 1;
                compare(tile.read(true), expected.back(), "final drained result");
            }
            uint64_t cycles = tile.cycles - begin;
            uint64_t bytes = tile.written_bytes + tile.read_bytes - begin_bytes;
            uint64_t expected_cycles = overlap ? loads + (count - 1) * (1 + std::max(loads, compute_cycles)) +
                                                1 + compute_cycles + elements
                                              : count * (loads + 1 + compute_cycles + elements);
            require(cycles == expected_cycles, "end-to-end streaming cycle count");
            require(bytes == count * (loads + elements) * 4, "end-to-end streaming byte count");
            std::cout << "BENCH design=" << design_name << " mode=" << (acc ? "bf16_accumulate" : "bf16_inputs")
                      << " overlap=" << overlap << " tiles=" << count << " cycles=" << cycles
                      << " bytes=" << bytes << " flops=" << count * tile_flops
                      << " target_mflops=" << double(count * tile_flops) / cycles * target_mhz << '\n';
        }
    }
    tile.dut.host_bank = tile.dut.job_bank = 0;
}
