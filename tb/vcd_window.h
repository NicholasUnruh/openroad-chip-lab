// VCD of one named benchmark for routed-netlist power (tools/tile_power.sh).
// Run a --trace build with +vcd=<file> +vcd_bench=<name>: the tests then run only their
// benchmarks and dump the named one, at the 40 MHz target, from its first clock to its last.
#pragma once
#include "verilated.h"
#include <cstdint>
#include <string>
#if VM_TRACE
#include "verilated_vcd_c.h"
#endif

class VcdWindow {
   public:
    static constexpr uint64_t period_ps = 25000;

    // True when the command line asks for a benchmark VCD.
    static bool requested() { return !arg("+vcd_bench=").empty(); }

    template <class Model>
    void begin(Model &dut, const std::string &bench, uint64_t cycle) {
#if VM_TRACE
        if (bench != arg("+vcd_bench=")) return;
        dut.trace(&file, 2);  // depth 2: each cell instance's pins, the names OpenSTA annotates
        file.set_time_unit("1ps");
        file.set_time_resolution("1ps");
        file.open(arg("+vcd=").c_str());
        first = cycle;
        on = true;
#else
        (void)dut, (void)bench, (void)cycle;
#endif
    }

    // Called with the cycle count before each clock: dumps the low phase and the high phase.
    void edge(uint64_t cycle, bool high) {
#if VM_TRACE
        if (on) file.dump((cycle - first) * period_ps + (high ? period_ps / 2 : 0));
#else
        (void)cycle, (void)high;
#endif
    }

    void end(uint64_t cycle) {
#if VM_TRACE
        if (!on) return;
        file.dump((cycle - first) * period_ps);
        file.close();
        on = false;
#else
        (void)cycle;
#endif
    }

   private:
    static std::string arg(const char *prefix) {
        std::string match = Verilated::commandArgsPlusMatch(prefix + 1);
        return match.size() > std::string(prefix).size() ? match.substr(std::string(prefix).size())
                                                         : std::string();
    }
#if VM_TRACE
    VerilatedVcdC file;
    uint64_t first = 0;
    bool on = false;
#endif
};
