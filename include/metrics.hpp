#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

inline float compute_rmse(const std::vector<float>& pred, const std::vector<float>& truth) {
    if (pred.size() != truth.size()) {
        throw std::runtime_error("RMSE: prediction and truth sizes differ");
    }
    if (pred.empty()) {
        throw std::runtime_error("RMSE: empty vectors");
    }

    double sum_sq = 0.0;
    for (std::size_t i = 0; i < pred.size(); ++i) {
        const double diff = static_cast<double>(pred[i]) - static_cast<double>(truth[i]);
        sum_sq += diff * diff;
    }

    return static_cast<float>(std::sqrt(sum_sq / static_cast<double>(pred.size())));
}
