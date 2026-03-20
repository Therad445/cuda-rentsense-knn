#pragma once

#include <vector>

struct CudaRunResult {
    std::vector<float> predictions;
    float total_ms = 0.0f;
    float h2d_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    float post_ms = 0.0f;
};

CudaRunResult knn_predict_cuda_naive(const std::vector<float>& X_train,
                                     const std::vector<float>& y_train,
                                     int train_rows,
                                     int cols,
                                     const std::vector<float>& X_query,
                                     int query_rows,
                                     int k);

CudaRunResult knn_predict_cuda_tiled(const std::vector<float>& X_train,
                                     const std::vector<float>& y_train,
                                     int train_rows,
                                     int cols,
                                     const std::vector<float>& X_query,
                                     int query_rows,
                                     int k);
