#include "knn_cpu.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

int checked_k(int k, int train_rows) {
    if (train_rows <= 0) {
        throw std::runtime_error("kNN: train_rows must be positive");
    }
    if (k <= 0) {
        throw std::runtime_error("kNN: k must be positive");
    }
    return std::min(k, train_rows);
}

float predict_one_from_row_distances(const float* row_dist,
                                     const std::vector<float>& y_train,
                                     int train_rows,
                                     int k,
                                     std::vector<int>& indices) {
    std::iota(indices.begin(), indices.end(), 0);
    const int eff_k = checked_k(k, train_rows);
    const int nth = eff_k - 1;

    std::nth_element(indices.begin(),
                     indices.begin() + nth,
                     indices.end(),
                     [&](int a, int b) {
                         return row_dist[a] < row_dist[b];
                     });

    float sum = 0.0f;
    for (int i = 0; i < eff_k; ++i) {
        sum += y_train[indices[i]];
    }
    return sum / static_cast<float>(eff_k);
}

}  // namespace

std::vector<float> knn_predict_cpu(const std::vector<float>& X_train,
                                   const std::vector<float>& y_train,
                                   int train_rows,
                                   int cols,
                                   const std::vector<float>& X_query,
                                   int query_rows,
                                   int k) {
    if (train_rows <= 0 || query_rows <= 0 || cols <= 0) {
        throw std::runtime_error("kNN CPU: invalid dimensions");
    }
    if (static_cast<int>(y_train.size()) != train_rows) {
        throw std::runtime_error("kNN CPU: y_train size mismatch");
    }
    if (static_cast<int>(X_train.size()) != train_rows * cols) {
        throw std::runtime_error("kNN CPU: X_train size mismatch");
    }
    if (static_cast<int>(X_query.size()) != query_rows * cols) {
        throw std::runtime_error("kNN CPU: X_query size mismatch");
    }

    std::vector<float> predictions(query_rows, 0.0f);
    std::vector<float> dist_row(train_rows, 0.0f);
    std::vector<int> indices(train_rows);

    const int eff_k = checked_k(k, train_rows);

    for (int q = 0; q < query_rows; ++q) {
        const float* qrow = X_query.data() + static_cast<std::size_t>(q) * cols;

        for (int n = 0; n < train_rows; ++n) {
            const float* trow = X_train.data() + static_cast<std::size_t>(n) * cols;
            float sum = 0.0f;
            for (int d = 0; d < cols; ++d) {
                const float diff = qrow[d] - trow[d];
                sum += diff * diff;
            }
            dist_row[n] = sum;
        }

        std::iota(indices.begin(), indices.end(), 0);
        const int nth = eff_k - 1;

        std::nth_element(indices.begin(),
                         indices.begin() + nth,
                         indices.end(),
                         [&](int a, int b) {
                             return dist_row[a] < dist_row[b];
                         });

        float y_sum = 0.0f;
        for (int i = 0; i < eff_k; ++i) {
            y_sum += y_train[indices[i]];
        }
        predictions[q] = y_sum / static_cast<float>(eff_k);
    }

    return predictions;
}

std::vector<float> knn_predict_from_distances_cpu(const std::vector<float>& dist,
                                                  const std::vector<float>& y_train,
                                                  int query_rows,
                                                  int train_rows,
                                                  int k) {
    if (query_rows <= 0 || train_rows <= 0) {
        throw std::runtime_error("kNN from distances: invalid dimensions");
    }
    if (static_cast<int>(dist.size()) != query_rows * train_rows) {
        throw std::runtime_error("kNN from distances: dist size mismatch");
    }
    if (static_cast<int>(y_train.size()) != train_rows) {
        throw std::runtime_error("kNN from distances: y_train size mismatch");
    }

    std::vector<float> predictions(query_rows, 0.0f);
    std::vector<int> indices(train_rows);

    for (int q = 0; q < query_rows; ++q) {
        const float* row_dist = dist.data() + static_cast<std::size_t>(q) * train_rows;
        predictions[q] = predict_one_from_row_distances(row_dist, y_train, train_rows, k, indices);
    }

    return predictions;
}
