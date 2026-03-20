#pragma once

#include <string>
#include <vector>

struct DataSet {
    int rows = 0;
    int cols = 0;
    std::vector<float> X;
    std::vector<float> y;
};

DataSet generate_synthetic_dataset(int rows, int cols, unsigned int seed);
DataSet load_binary_dataset(const std::string& features_path,
                            const std::string& labels_path);
