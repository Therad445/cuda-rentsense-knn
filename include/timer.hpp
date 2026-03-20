#pragma once

#include <chrono>

class HostTimer {
public:
    using clock = std::chrono::high_resolution_clock;

    void start() {
        start_ = clock::now();
    }

    double elapsed_ms() const {
        const auto end = clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    clock::time_point start_{};
};
