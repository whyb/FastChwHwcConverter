#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace whyb_test {

// Fixed seed keeps benchmark/validation runs reproducible.
inline std::mt19937& rng() {
    static std::mt19937 gen(0x5EED1234u);
    return gen;
}

inline std::vector<uint8_t> random_u8(const size_t count) {
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<uint8_t> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = static_cast<uint8_t>(dist(rng()));
    }
    return data;
}

inline std::vector<float> random_f32(const size_t count, const float lo, const float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng());
    }
    return data;
}

/**
 * Runs `fn()` for `warmup` calls, then `reps` timed batches of `iterations`
 * calls each. Returns the minimum batch average in microseconds per call.
 * Taking the minimum across repetitions reduces noise from the OS / other
 * cores, and the warm-up calls let caches, clocks and GPU pipelines settle.
 */
template <typename Fn>
double measure_min(Fn&& fn, const int warmup, const int reps, const int iterations) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    double best_us = std::numeric_limits<double>::max();
    for (int r = 0; r < reps; ++r) {
        const auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            fn();
        }
        const auto end = std::chrono::high_resolution_clock::now();
        const double avg_us =
            std::chrono::duration<double, std::micro>(end - start).count() / static_cast<double>(iterations);
        best_us = std::min(best_us, avg_us);
    }
    return best_us; // microseconds
}

} // namespace whyb_test
