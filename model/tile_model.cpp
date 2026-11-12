// Cycle-level performance model of training_tile_v4 and training_tile_v6 (docs/tile-perf-model.md).
//
// V4: the tile's busy/done timing (issue cycles + 3-clock FMA drain) and its host rules: one write
// and one read transfer per clock, writes only to a buffer the tile is not computing on, and an
// accepted start wins over a simultaneous write. The host is greedy: it writes jobs in order as
// soon as a buffer frees and reads results as soon as a job is done.
// V6: the GEMM sequencer's control registers transliterated clock for clock from
// rtl/training_tile_v6.v (loader, waiting job, engine, dot-product pipeline, C-bank drain),
// driven by the host loop of tb/scratchpad_tile_test.cpp. Arithmetic is not modelled.
//
// usage: build/tile_model bench | sweep
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr uint64_t max_cycles = 1u << 22;  // bound on every simulated run
constexpr double target_mhz = 40;

void require(bool condition, const char *what) {
    if (!condition) {
        std::fprintf(stderr, "model invariant failed: %s\n", what);
        std::exit(1);
    }
}


// ---------------------------------------------------------------- V4

struct V4Config {
    unsigned lanes = 8, buffers = 2, port_bytes = 4;  // port_bytes per clock, each direction
    unsigned compute_cycles() const { return 512 / lanes + 3; }
};

struct V4Job {
    unsigned ab_bytes, c_bytes, out_bytes;  // operand writes, FP32 C writes, result reads
    bool accumulate;                       // retain C; otherwise start clears it
};

struct V4Result { uint64_t cycles, bytes; };

struct V4Progress {
    std::vector<unsigned> written, read;
    std::vector<bool> done;
};

bool v4_inputs_written(const V4Job &job, unsigned written) { return written >= job.ab_bytes + job.c_bytes; }

// Earliest job in `jobs` whose outputs are not fully read, or jobs.size().
unsigned v4_first_unread(const std::vector<V4Job> &jobs, const V4Progress &p, unsigned limit) {
    for (unsigned j = 0; j < limit; ++j)
        if (p.read[j] < jobs[j].out_bytes) return j;
    return limit;
}

// May the host write job w's next bytes this clock?
bool v4_can_write(const V4Config &cfg, const std::vector<V4Job> &jobs, const V4Progress &p, unsigned w,
                  bool busy, unsigned active, bool overlap) {
    if (w >= jobs.size()) return false;
    unsigned buffer = w % cfg.buffers;
    if (busy && active % cfg.buffers == buffer) return false;
    if (w >= cfg.buffers && !p.done[w - cfg.buffers]) return false;  // previous user of the buffer
    bool writing_c = p.written[w] >= jobs[w].ab_bytes;
    if (writing_c && w >= cfg.buffers && p.read[w - cfg.buffers] < jobs[w - cfg.buffers].out_bytes)
        return false;  // C overwrite waits for the old results
    if (!overlap) return !busy && v4_first_unread(jobs, p, w) == w;
    return true;
}

V4Result run_v4(const V4Config &cfg, const std::vector<V4Job> &jobs, bool overlap) {
    require(cfg.lanes && 512 % cfg.lanes == 0 && cfg.buffers && cfg.port_bytes, "V4 configuration");
    const unsigned n = unsigned(jobs.size());
    V4Progress p{std::vector<unsigned>(n), std::vector<unsigned>(n), std::vector<bool>(n)};
    bool busy = false;
    unsigned active = 0, remaining = 0, next_start = 0, w = 0;
    uint64_t cycle = 0, bytes = 0;
    while (v4_first_unread(jobs, p, n) < n || next_start < n || busy) {
        require(++cycle < max_cycles, "V4 run bounded");
        unsigned s = next_start;
        bool start = !busy && s < n && v4_inputs_written(jobs[s], p.written[s]) &&
                     (jobs[s].accumulate || s < cfg.buffers ||
                      p.read[s - cfg.buffers] >= jobs[s - cfg.buffers].out_bytes) &&
                     (overlap || v4_first_unread(jobs, p, s) == s);
        if (!start && v4_can_write(cfg, jobs, p, w, busy, active, overlap)) {
            unsigned limit = p.written[w] < jobs[w].ab_bytes ? jobs[w].ab_bytes : jobs[w].ab_bytes + jobs[w].c_bytes;
            unsigned moved = std::min(cfg.port_bytes, limit - p.written[w]);
            p.written[w] += moved;
            bytes += moved;
            if (v4_inputs_written(jobs[w], p.written[w])) ++w;
        }
        unsigned r = v4_first_unread(jobs, p, n);
        if (r < n && p.done[r]) {
            unsigned moved = std::min(cfg.port_bytes, jobs[r].out_bytes - p.read[r]);
            p.read[r] += moved;
            bytes += moved;
        }
        if (start) {
            busy = true;
            active = s;
            remaining = cfg.compute_cycles();
            ++next_start;
        } else if (busy && --remaining == 0) {
            busy = false;
            p.done[active] = true;
        }
    }
    return {cycle, bytes};
}

// Streamed GEMM of `count` size^3 products in 8x8x8 jobs. Each group of `buffers` output blocks
// runs its K chunks round-robin so every buffer retains its block's C (tb/bf16_tile_tests.h).
std::vector<V4Job> v4_gemm_jobs(unsigned count, unsigned size, unsigned buffers) {
    unsigned blocks = size / 8, outputs = blocks * blocks;
    std::vector<V4Job> jobs;
    for (unsigned m = 0; m < count; ++m)
        for (unsigned group = 0; group < outputs; group += buffers)
            for (unsigned k = 0; k < blocks; ++k)
                for (unsigned slot = 0; slot < buffers && group + slot < outputs; ++slot)
                    jobs.push_back({256, 0, k == blocks - 1 ? 256u : 0u, k != 0});
    return jobs;
}

// ---------------------------------------------------------------- V6

struct V6Config {
    unsigned units = 4, c_banks = 2, port_bytes = 4;
    static constexpr unsigned lat = 6;      // dot-product latency
    unsigned run_len() const { return 64 / units; }  // clocks per 8x8x8 job
};

struct V6Gemm { unsigned m, n, k; bool ta; };  // tiles per dimension

class V6Tile {
public:
    explicit V6Tile(const V6Config &c) : cfg(c) { require(c.c_banks >= 1 && c.c_banks <= 8, "C banks"); }
    bool busy = false, done = false;
    bool result_valid() const { return queued > 0; }
    void tick(bool start, const V6Gemm &g, bool ack);

private:
    V6Config cfg;
    unsigned m_max = 0, n_max = 0, k_max = 0;
    bool ta = false;
    bool ld_valid = false;
    unsigned ld_cbank = 0, ld_i = 0, ld_j = 0, ld_k = 0;
    bool nxt_valid = false, nxt_k0 = false, nxt_last = false;
    unsigned nxt_cbank = 0;
    bool load_active = false, loaded = false, wr_valid = false;
    unsigned load_index = 0;
    bool run_active = false, run_last = false;
    unsigned run_cbank = 0, run_cycle = 0;
    std::array<bool, V6Config::lat> p_valid{}, p_last{};
    std::array<unsigned, V6Config::lat> p_cbank{};
    unsigned pending = 0, inflight = 0;  // bank bitmasks: finished awaiting drain; last job issued
    std::array<unsigned, 8> fifo{};      // drain order of pending banks
    unsigned queued = 0;
    void advance_loader();
    void step_pipeline(bool ack);
};

void V6Tile::advance_loader() {
    if (ld_k != k_max) { ++ld_k; return; }
    ld_k = 0;
    ld_cbank = (ld_cbank + 1) % cfg.c_banks;
    if (ld_j != n_max) { ++ld_j; return; }
    ld_j = 0;
    if (ld_i == m_max) ld_valid = false;
    ++ld_i;
}

// Pipeline shift, writeback push and host pop; the push precedes the pop in FIFO order.
void V6Tile::step_pipeline(bool ack) {
    bool push = p_valid[cfg.lat - 1] && p_last[cfg.lat - 1];
    unsigned push_bank = p_cbank[cfg.lat - 1];
    bool pop = ack && queued > 0;
    for (unsigned s = cfg.lat - 1; s > 0; --s) {
        p_valid[s] = p_valid[s - 1];
        p_last[s] = p_last[s - 1];
        p_cbank[s] = p_cbank[s - 1];
    }
    p_valid[0] = run_active;
    p_last[0] = run_active && run_last && run_cycle == cfg.run_len() - 1;
    p_cbank[0] = run_cbank;
    if (pop) {
        pending &= ~(1u << fifo[0]);
        for (unsigned q = 1; q < queued; ++q) fifo[q - 1] = fifo[q];
        --queued;
    }
    if (push) {
        require(!(pending >> push_bank & 1u), "C bank overwritten before drain");
        pending |= 1u << push_bank;
        inflight &= ~(1u << push_bank);
        fifo[queued++] = push_bank;
    }
}

void V6Tile::tick(bool start, const V6Gemm &g, bool ack) {
    const unsigned last_index = 15;  // 16 scratchpad rows per job: 8 of B, then 8 of A
    bool nxt_ready = loaded || (load_active ? !ta && load_index >= 9 : wr_valid);
    bool slot_free = !run_active || run_cycle == cfg.run_len() - 1;
    bool eng_start = nxt_valid && nxt_ready && slot_free &&
                     !(nxt_k0 && ((pending | inflight) >> nxt_cbank & 1u));
    bool load_go = busy && ld_valid && !nxt_valid && (!load_active || load_index == last_index);
    bool any_p = std::any_of(p_valid.begin(), p_valid.end(), [](bool v) { return v; });
    done = false;
    if (!busy) {
        if (start) {
            busy = true;
            m_max = g.m - 1; n_max = g.n - 1; k_max = g.k - 1; ta = g.ta;
            ld_valid = true; ld_cbank = 0; ld_i = ld_j = ld_k = 0;
        }
    } else if (!ld_valid && !nxt_valid && !run_active && !any_p && pending == 0) {
        busy = false;
        done = true;
    }
    bool was_load_active = load_active, was_wr_valid = wr_valid;
    wr_valid = load_active;
    if (load_go) {
        load_active = true; load_index = 0; loaded = false;
        nxt_valid = true; nxt_cbank = ld_cbank; nxt_k0 = ld_k == 0; nxt_last = ld_k == k_max;
        advance_loader();
    } else if (load_active) {
        if (load_index == last_index) load_active = false;
        ++load_index;
    }
    if (was_wr_valid && !was_load_active && !load_go) loaded = true;
    step_pipeline(ack);
    if (run_active) {
        if (run_cycle == cfg.run_len() - 1) run_active = false;
        ++run_cycle;
    }
    if (eng_start) {
        nxt_valid = false;
        run_active = true; run_cycle = 0; run_cbank = nxt_cbank; run_last = nxt_last;
        if (nxt_last) inflight |= 1u << nxt_cbank;
    }
}

// Host side of tb/scratchpad_tile_test.cpp: stream queued writes, drain every 256-byte output
// tile at port_bytes per clock, release it on the clock of its last read.
struct V6Host {
    V6Config cfg;
    V6Tile tile{cfg};
    uint64_t cycles = 0, bytes = 0, write_backlog = 0;
    void tick(bool start, const V6Gemm &g, bool ack) {
        require(++cycles < max_cycles, "V6 run bounded");
        tile.tick(start, g, ack);
    }
    bool drive_write() {
        if (!write_backlog) return false;
        uint64_t moved = std::min<uint64_t>(cfg.port_bytes, write_backlog);
        write_backlog -= moved;
        bytes += moved;
        return true;
    }
    void store(uint64_t b) {
        write_backlog += b;
        while (drive_write()) tick(false, {}, false);
    }
    void run(const V6Gemm &g) {
        tick(true, g, false);
        unsigned count = 0;
        for (uint64_t guard = 0;; ++guard) {
            require(guard < max_cycles, "V6 GEMM bounded");
            bool written = drive_write();
            (void)written;  // write timing never stalls the tile; only the backlog matters
            bool ack = false;
            if (tile.result_valid()) {
                unsigned moved = std::min(cfg.port_bytes, 256 - count);
                count += moved;
                bytes += moved;
                if (count == 256) { ack = true; count = 0; }
            }
            tick(false, g, ack);
            if (tile.done) break;
        }
        tick(false, g, false);
    }
};

uint64_t operand_bytes(const V6Gemm &g) { return uint64_t(g.m * g.k + g.k * g.n) * 64 * 2; }

V4Result v6_gemm32(const V6Config &cfg) {
    V6Host host{cfg};
    V6Gemm g{4, 4, 4, false};
    host.store(operand_bytes(g));
    host.run(g);
    return {host.cycles, host.bytes};
}

// Eight 16^3 GEMMs; each one's successor loads into the other scratchpad half while it runs.
V4Result v6_gemm16_pipelined(const V6Config &cfg) {
    V6Host host{cfg};
    V6Gemm g{2, 2, 2, false};
    host.store(operand_bytes(g));
    for (unsigned n = 0; n < 8; ++n) {
        if (n + 1 < 8) host.write_backlog += operand_bytes(g);
        host.run(g);
        while (host.drive_write()) host.tick(false, g, false);
    }
    return {host.cycles, host.bytes};
}

// ---------------------------------------------------------------- reports

void bench_line(const char *design, const char *mode, const char *extra, V4Result r, unsigned flops) {
    std::printf("BENCH design=%s mode=%s%s cycles=%llu bytes=%llu flops=%u target_mflops=%g\n", design, mode,
                extra, (unsigned long long)r.cycles, (unsigned long long)r.bytes, flops,
                double(flops) / double(r.cycles) * target_mhz);
}

void bench() {
    V4Config v4;
    for (bool acc : {false, true})
        for (bool overlap : {false, true}) {
            std::vector<V4Job> jobs(64, V4Job{256, acc ? 256u : 0u, 256, acc});
            V4Result r = run_v4(v4, jobs, overlap);
            std::printf("BENCH design=training_tile_v4 mode=%s overlap=%d tiles=64 cycles=%llu bytes=%llu "
                        "flops=65536 target_mflops=%g\n", acc ? "bf16_accumulate" : "bf16_inputs", int(overlap),
                        (unsigned long long)r.cycles, (unsigned long long)r.bytes, 65536.0 / double(r.cycles) * target_mhz);
        }
    const unsigned shapes[3][2] = {{64, 8}, {8, 16}, {1, 32}};
    for (const auto &s : shapes) {
        std::vector<V4Job> jobs = v4_gemm_jobs(s[0], s[1], v4.buffers);
        char mode[32], extra[48];
        std::snprintf(mode, sizeof mode, "gemm%u_stream", s[1]);
        std::snprintf(extra, sizeof extra, " matrices=%u jobs=%zu", s[0], jobs.size());
        bench_line("training_tile_v4", mode, extra, run_v4(v4, jobs, true), 65536);
    }
    V6Config v6;
    bench_line("training_tile_v6", "gemm32_scratchpad", "", v6_gemm32(v6), 65536);
    bench_line("training_tile_v6", "gemm16_pipelined", " matrices=8", v6_gemm16_pipelined(v6), 65536);
}

void sweep() {
    std::printf("V4 32^3 GEMM, streamed (cycles; MACs per clock)\n");
    std::printf("lanes buffers port_bytes cycles macs_per_clock lane_util\n");
    for (unsigned lanes : {4u, 8u, 16u, 32u, 64u})
        for (unsigned buffers : {1u, 2u, 3u})
            for (unsigned port : {2u, 4u, 8u, 16u}) {
                V4Config c{lanes, buffers, port};
                V4Result r = run_v4(c, v4_gemm_jobs(1, 32, buffers), true);
                double macs = 32768.0 / double(r.cycles);
                std::printf("%u %u %u %llu %.2f %.3f\n", lanes, buffers, port, (unsigned long long)r.cycles, macs,
                            macs / lanes);
            }
    std::printf("V6 (cycles: 32^3 with fill, eight pipelined 16^3)\n");
    std::printf("units c_banks port_bytes gemm32 gemm16x8 macs_per_clock_16x8 unit_util_16x8\n");
    for (unsigned units : {2u, 4u, 8u})
        for (unsigned banks : {1u, 2u, 3u})
            for (unsigned port : {2u, 4u, 8u, 16u}) {
                V6Config c{units, banks, port};
                V4Result a = v6_gemm32(c), b = v6_gemm16_pipelined(c);
                double macs = 32768.0 / double(b.cycles);
                std::printf("%u %u %u %llu %llu %.2f %.3f\n", units, banks, port, (unsigned long long)a.cycles,
                            (unsigned long long)b.cycles, macs, macs / (8.0 * units));
            }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::strcmp(argv[1], "bench") == 0) bench();
    else if (argc == 2 && std::strcmp(argv[1], "sweep") == 0) sweep();
    else {
        std::fprintf(stderr, "usage: tile_model bench | sweep\n");
        return 2;
    }
    return 0;
}
