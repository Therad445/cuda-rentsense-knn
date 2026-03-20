#include "knn_cuda.cuh"

#include "knn_cpu.hpp"
#include "timer.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define CUDA_CHECK(call)                                                             \
    do {                                                                             \
        cudaError_t err__ = (call);                                                  \
        if (err__ != cudaSuccess) {                                                  \
            throw std::runtime_error(std::string("CUDA error: ") +                   \
                                     cudaGetErrorString(err__) +                     \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                            \
    } while (0)

constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;
constexpr int TILE_K = 16;

__global__ void pairwise_distance_naive_kernel(const float* __restrict__ Xq,
                                               const float* __restrict__ Xt,
                                               float* __restrict__ dist,
                                               int query_rows,
                                               int train_rows,
                                               int cols) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int q = blockIdx.y * blockDim.y + threadIdx.y;

    if (q < query_rows && n < train_rows) {
        float sum = 0.0f;
        const int q_base = q * cols;
        const int t_base = n * cols;

        for (int d = 0; d < cols; ++d) {
            const float diff = Xq[q_base + d] - Xt[t_base + d];
            sum += diff * diff;
        }

        dist[static_cast<std::size_t>(q) * train_rows + n] = sum;
    }
}

__global__ void pairwise_distance_tiled_kernel(const float* __restrict__ Xq,
                                               const float* __restrict__ Xt,
                                               float* __restrict__ dist,
                                               int query_rows,
                                               int train_rows,
                                               int cols) {
    __shared__ float sh_query[BLOCK_Y][TILE_K];
    __shared__ float sh_train[BLOCK_X][TILE_K];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int n = blockIdx.x * blockDim.x + tx;
    const int q = blockIdx.y * blockDim.y + ty;
    const int linear_tid = ty * blockDim.x + tx;

    float sum = 0.0f;

    for (int k0 = 0; k0 < cols; k0 += TILE_K) {
        for (int idx = linear_tid; idx < BLOCK_Y * TILE_K; idx += blockDim.x * blockDim.y) {
            const int local_q = idx / TILE_K;
            const int local_k = idx % TILE_K;
            const int q_idx = blockIdx.y * blockDim.y + local_q;
            const int d = k0 + local_k;

            sh_query[local_q][local_k] =
                (q_idx < query_rows && d < cols)
                    ? Xq[static_cast<std::size_t>(q_idx) * cols + d]
                    : 0.0f;
        }

        for (int idx = linear_tid; idx < BLOCK_X * TILE_K; idx += blockDim.x * blockDim.y) {
            const int local_n = idx / TILE_K;
            const int local_k = idx % TILE_K;
            const int n_idx = blockIdx.x * blockDim.x + local_n;
            const int d = k0 + local_k;

            sh_train[local_n][local_k] =
                (n_idx < train_rows && d < cols)
                    ? Xt[static_cast<std::size_t>(n_idx) * cols + d]
                    : 0.0f;
        }

        __syncthreads();

        if (q < query_rows && n < train_rows) {
            const int limit = min(TILE_K, cols - k0);
            for (int kk = 0; kk < limit; ++kk) {
                const float diff = sh_query[ty][kk] - sh_train[tx][kk];
                sum += diff * diff;
            }
        }

        __syncthreads();
    }

    if (q < query_rows && n < train_rows) {
        dist[static_cast<std::size_t>(q) * train_rows + n] = sum;
    }
}

CudaRunResult run_cuda_knn(const std::vector<float>& X_train,
                           const std::vector<float>& y_train,
                           int train_rows,
                           int cols,
                           const std::vector<float>& X_query,
                           int query_rows,
                           int k,
                           bool tiled) {
    if (train_rows <= 0 || query_rows <= 0 || cols <= 0) {
        throw std::runtime_error("CUDA kNN: invalid dimensions");
    }
    if (static_cast<int>(X_train.size()) != train_rows * cols) {
        throw std::runtime_error("CUDA kNN: X_train size mismatch");
    }
    if (static_cast<int>(X_query.size()) != query_rows * cols) {
        throw std::runtime_error("CUDA kNN: X_query size mismatch");
    }
    if (static_cast<int>(y_train.size()) != train_rows) {
        throw std::runtime_error("CUDA kNN: y_train size mismatch");
    }

    CudaRunResult result;
    HostTimer total_timer;
    total_timer.start();

    const std::size_t train_bytes = X_train.size() * sizeof(float);
    const std::size_t query_bytes = X_query.size() * sizeof(float);
    const std::size_t dist_elems = static_cast<std::size_t>(query_rows) * train_rows;
    const std::size_t dist_bytes = dist_elems * sizeof(float);

    float* d_Xt = nullptr;
    float* d_Xq = nullptr;
    float* d_dist = nullptr;

    try {
        CUDA_CHECK(cudaMalloc(&d_Xt, train_bytes));
        CUDA_CHECK(cudaMalloc(&d_Xq, query_bytes));
        CUDA_CHECK(cudaMalloc(&d_dist, dist_bytes));

        HostTimer h2d_timer;
        h2d_timer.start();
        CUDA_CHECK(cudaMemcpy(d_Xt, X_train.data(), train_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_Xq, X_query.data(), query_bytes, cudaMemcpyHostToDevice));
        result.h2d_ms = static_cast<float>(h2d_timer.elapsed_ms());

        dim3 block(BLOCK_X, BLOCK_Y);
        dim3 grid((train_rows + BLOCK_X - 1) / BLOCK_X,
                  (query_rows + BLOCK_Y - 1) / BLOCK_Y);

        cudaEvent_t start_ev{};
        cudaEvent_t stop_ev{};
        CUDA_CHECK(cudaEventCreate(&start_ev));
        CUDA_CHECK(cudaEventCreate(&stop_ev));

        CUDA_CHECK(cudaEventRecord(start_ev));
        if (tiled) {
            pairwise_distance_tiled_kernel<<<grid, block>>>(d_Xq, d_Xt, d_dist,
                                                            query_rows, train_rows, cols);
        } else {
            pairwise_distance_naive_kernel<<<grid, block>>>(d_Xq, d_Xt, d_dist,
                                                            query_rows, train_rows, cols);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop_ev));
        CUDA_CHECK(cudaEventSynchronize(stop_ev));
        CUDA_CHECK(cudaEventElapsedTime(&result.kernel_ms, start_ev, stop_ev));

        CUDA_CHECK(cudaEventDestroy(start_ev));
        CUDA_CHECK(cudaEventDestroy(stop_ev));

        std::vector<float> dist_host(dist_elems);

        HostTimer d2h_timer;
        d2h_timer.start();
        CUDA_CHECK(cudaMemcpy(dist_host.data(), d_dist, dist_bytes, cudaMemcpyDeviceToHost));
        result.d2h_ms = static_cast<float>(d2h_timer.elapsed_ms());

        HostTimer post_timer;
        post_timer.start();
        result.predictions = knn_predict_from_distances_cpu(dist_host, y_train, query_rows, train_rows, k);
        result.post_ms = static_cast<float>(post_timer.elapsed_ms());

        result.total_ms = static_cast<float>(total_timer.elapsed_ms());

        CUDA_CHECK(cudaFree(d_Xt));
        CUDA_CHECK(cudaFree(d_Xq));
        CUDA_CHECK(cudaFree(d_dist));
    } catch (...) {
        if (d_Xt) {
            cudaFree(d_Xt);
        }
        if (d_Xq) {
            cudaFree(d_Xq);
        }
        if (d_dist) {
            cudaFree(d_dist);
        }
        throw;
    }

    return result;
}

}  // namespace

CudaRunResult knn_predict_cuda_naive(const std::vector<float>& X_train,
                                     const std::vector<float>& y_train,
                                     int train_rows,
                                     int cols,
                                     const std::vector<float>& X_query,
                                     int query_rows,
                                     int k) {
    return run_cuda_knn(X_train, y_train, train_rows, cols, X_query, query_rows, k, false);
}

CudaRunResult knn_predict_cuda_tiled(const std::vector<float>& X_train,
                                     const std::vector<float>& y_train,
                                     int train_rows,
                                     int cols,
                                     const std::vector<float>& X_query,
                                     int query_rows,
                                     int k) {
    return run_cuda_knn(X_train, y_train, train_rows, cols, X_query, query_rows, k, true);
}
