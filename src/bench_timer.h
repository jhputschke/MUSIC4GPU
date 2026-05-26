// Lightweight RAII profiler — header-only, opt-in via MUSIC_PROFILE=1.
//
// Usage:
//   #include "bench_timer.h"
//   { bench::Timer t("some_section"); do_work(); }   // accumulates time
//   bench::dump();                                   // print totals (e.g. at end of EvolveIt)
//
// When MUSIC_PROFILE is not set, the destructor short-circuits and the
// overhead is two std::chrono::steady_clock::now() calls per scope (~100 ns)
// plus one branch — negligible for the millisecond-scale sections we time.
//
// Not thread-safe: only call bench::Timer from the main thread (i.e. outside
// OpenMP parallel regions).  Used here only at function-call granularity.

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace bench {

inline bool enabled() {
    static bool e = []() {
        const char* v = std::getenv("MUSIC_PROFILE");
        return v && v[0] == '1';
    }();
    return e;
}

struct Acc {
    double seconds = 0.0;
    long   count   = 0;
};

inline std::map<std::string, Acc>& accumulators() {
    static std::map<std::string, Acc> m;
    return m;
}

inline void add(const char* name, double seconds) {
    auto& a = accumulators()[name];
    a.seconds += seconds;
    a.count   += 1;
}

class Timer {
public:
    explicit Timer(const char* name)
        : name_(name), t0_(std::chrono::steady_clock::now()) {}
    ~Timer() {
        if (!enabled()) return;
        auto dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0_).count();
        add(name_, dt);
    }
private:
    const char* name_;
    std::chrono::steady_clock::time_point t0_;
};

inline void dump() {
    if (!enabled()) return;
    auto& m = accumulators();
    std::vector<std::pair<std::string, Acc>> entries(m.begin(), m.end());
    std::sort(entries.begin(), entries.end(),
        [](const std::pair<std::string, Acc>& a,
           const std::pair<std::string, Acc>& b) {
            return a.second.seconds > b.second.seconds;
        });

    std::fprintf(stderr,
        "\n[MUSIC-PROFILE] timer breakdown (sorted by total time)\n");
    std::fprintf(stderr,
        "  %-40s  %10s  %8s  %12s\n",
        "Section", "total (s)", "calls", "avg (ms)");
    std::fprintf(stderr,
        "  %-40s  %10s  %8s  %12s\n",
        "----------------------------------------",
        "---------", "-----", "--------");
    for (const auto& kv : entries) {
        const auto& a = kv.second;
        const double avg_ms = a.seconds * 1000.0 / std::max(1L, a.count);
        std::fprintf(stderr,
            "  %-40s  %10.4f  %8ld  %12.4f\n",
            kv.first.c_str(), a.seconds, a.count, avg_ms);
    }
    std::fflush(stderr);
}

}  // namespace bench
