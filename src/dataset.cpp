#include "dataset.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

template <typename T>
void read_exact(std::ifstream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) {
        throw std::runtime_error("Failed to read binary file");
    }
}

std::vector<float> read_float_vector(std::ifstream& in, std::size_t count) {
    std::vector<float> data(count);
    in.read(reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(count * sizeof(float)));
    if (!in) {
        throw std::runtime_error("Failed to read float payload from binary file");
    }
    return data;
}

float synthetic_target(const float* row, int cols, std::mt19937& rng) {
    std::normal_distribution<float> noise(0.0f, 0.03f);

    float y = 0.0f;
    for (int d = 0; d < cols; ++d) {
        const float weight = 0.15f + 0.03f * static_cast<float>((d % 7) + 1);
        y += weight * row[d];
    }

    if (cols >= 2) {
        y += 0.35f * row[0] * row[1];
    }
    if (cols >= 3) {
        y += 0.12f * std::sin(3.0f * row[2]);
    }
    if (cols >= 4) {
        y += 0.08f * row[3] * row[3];
    }

    y += noise(rng);
    return y;
}

}  // namespace

DataSet generate_synthetic_dataset(int rows, int cols, unsigned int seed) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Synthetic dataset dimensions must be positive");
    }

    DataSet ds;
    ds.rows = rows;
    ds.cols = cols;
    ds.X.resize(static_cast<std::size_t>(rows) * cols);
    ds.y.resize(rows);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unif(0.0f, 1.0f);

    for (int i = 0; i < rows; ++i) {
        float* row_ptr = ds.X.data() + static_cast<std::size_t>(i) * cols;
        for (int d = 0; d < cols; ++d) {
            row_ptr[d] = unif(rng);
        }
        ds.y[i] = synthetic_target(row_ptr, cols, rng);
    }

    return ds;
}

DataSet load_binary_dataset(const std::string& features_path,
                            const std::string& labels_path) {
    std::ifstream xf(features_path, std::ios::binary);
    if (!xf) {
        throw std::runtime_error("Cannot open features file: " + features_path);
    }

    std::ifstream yf(labels_path, std::ios::binary);
    if (!yf) {
        throw std::runtime_error("Cannot open labels file: " + labels_path);
    }

    std::int32_t rows = 0;
    std::int32_t cols = 0;
    read_exact(xf, rows);
    read_exact(xf, cols);

    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Invalid matrix shape in features file");
    }

    DataSet ds;
    ds.rows = static_cast<int>(rows);
    ds.cols = static_cast<int>(cols);
    ds.X = read_float_vector(xf, static_cast<std::size_t>(rows) * cols);

    std::int32_t label_rows = 0;
    read_exact(yf, label_rows);
    if (label_rows != rows) {
        throw std::runtime_error("Label count does not match feature rows");
    }

    ds.y = read_float_vector(yf, static_cast<std::size_t>(label_rows));
    return ds;
}
