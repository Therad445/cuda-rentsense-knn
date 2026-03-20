#include "dataset.hpp"
#include "knn_cpu.hpp"
#include "knn_cuda.cuh"
#include "metrics.hpp"
#include "timer.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct Config {
    std::string mode = "all";
    std::string data = "synthetic";

    int N = 20000;
    int Q = 1024;
    int D = 32;
    int k = 5;
    unsigned int seed = 42;

    std::string train_features;
    std::string train_labels;
    std::string query_features;
    std::string query_labels;

    std::string csv_out;
};

struct RunSummary {
    std::string mode;
    float total_ms = 0.0f;
    float h2d_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    float post_ms = 0.0f;
    float rmse = std::numeric_limits<float>::quiet_NaN();
};

namespace {

void print_help() {
    std::cout << R"(Usage:
  ./cuda_rentsense_knn [options]

Modes:
  --mode cpu|naive|optimized|all

Data:
  --data synthetic
  --data binary --train_features path --train_labels path --query_features path --query_labels path

Synthetic options:
  --N <int>       train rows
  --Q <int>       query rows
  --D <int>       feature dimension
  --k <int>       neighbors
  --seed <int>

Optional:
  --csv <path>    append benchmark rows to CSV

Examples:
  ./cuda_rentsense_knn --mode all --data synthetic --N 10000 --Q 512 --D 32 --k 5
  ./cuda_rentsense_knn --mode optimized --data binary \
      --train_features data/processed/X_train.bin \
      --train_labels data/processed/y_train.bin \
      --query_features data/processed/X_query.bin \
      --query_labels data/processed/y_query.bin
)";
}

std::string require_value(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value after argument: ") + argv[i]);
    }
    return argv[++i];
}

Config parse_args(int argc, char** argv) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_help();
            std::exit(0);
        } else if (arg == "--mode") {
            cfg.mode = require_value(i, argc, argv);
        } else if (arg == "--data") {
            cfg.data = require_value(i, argc, argv);
        } else if (arg == "--N") {
            cfg.N = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--Q") {
            cfg.Q = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--D") {
            cfg.D = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--k") {
            cfg.k = std::stoi(require_value(i, argc, argv));
        } else if (arg == "--seed") {
            cfg.seed = static_cast<unsigned int>(std::stoul(require_value(i, argc, argv)));
        } else if (arg == "--train_features") {
            cfg.train_features = require_value(i, argc, argv);
        } else if (arg == "--train_labels") {
            cfg.train_labels = require_value(i, argc, argv);
        } else if (arg == "--query_features") {
            cfg.query_features = require_value(i, argc, argv);
        } else if (arg == "--query_labels") {
            cfg.query_labels = require_value(i, argc, argv);
        } else if (arg == "--csv") {
            cfg.csv_out = require_value(i, argc, argv);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    return cfg;
}

void load_data(const Config& cfg, DataSet& train, DataSet& query) {
    if (cfg.data == "synthetic") {
        train = generate_synthetic_dataset(cfg.N, cfg.D, cfg.seed);
        query = generate_synthetic_dataset(cfg.Q, cfg.D, cfg.seed + 1u);
        return;
    }

    if (cfg.data == "binary") {
        if (cfg.train_features.empty() || cfg.train_labels.empty() ||
            cfg.query_features.empty() || cfg.query_labels.empty()) {
            throw std::runtime_error("Binary mode requires train/query feature and label files");
        }

        train = load_binary_dataset(cfg.train_features, cfg.train_labels);
        query = load_binary_dataset(cfg.query_features, cfg.query_labels);

        if (train.cols != query.cols) {
            throw std::runtime_error("Train and query feature dimensions differ");
        }
        return;
    }

    throw std::runtime_error("Unknown data mode: " + cfg.data);
}

void append_csv_row(const std::string& path,
                    const Config& cfg,
                    int train_rows,
                    int query_rows,
                    int cols,
                    const RunSummary& r) {
    namespace fs = std::filesystem;
    const bool write_header = !fs::exists(path);

    if (const auto parent = fs::path(path).parent_path(); !parent.empty()) {
        fs::create_directories(parent);
    }

    std::ofstream out(path, std::ios::app);
    if (!out) {
        throw std::runtime_error("Cannot open CSV file for writing: " + path);
    }

    if (write_header) {
        out << "mode,data,N,Q,D,k,total_ms,h2d_ms,kernel_ms,d2h_ms,post_ms,rmse\n";
    }

    out << r.mode << ','
        << cfg.data << ','
        << train_rows << ','
        << query_rows << ','
        << cols << ','
        << cfg.k << ','
        << r.total_ms << ','
        << r.h2d_ms << ','
        << r.kernel_ms << ','
        << r.d2h_ms << ','
        << r.post_ms << ','
        << r.rmse << '\n';
}

void print_summary(const RunSummary& r) {
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\n[" << r.mode << "]\n";
    std::cout << "  total_ms  = " << r.total_ms << '\n';
    std::cout << "  h2d_ms    = " << r.h2d_ms << '\n';
    std::cout << "  kernel_ms = " << r.kernel_ms << '\n';
    std::cout << "  d2h_ms    = " << r.d2h_ms << '\n';
    std::cout << "  post_ms   = " << r.post_ms << '\n';
    std::cout << "  rmse      = " << r.rmse << '\n';
}

RunSummary run_cpu(const DataSet& train, const DataSet& query, int k) {
    HostTimer timer;
    timer.start();

    const auto pred = knn_predict_cpu(train.X, train.y, train.rows, train.cols, query.X, query.rows, k);

    RunSummary r;
    r.mode = "cpu";
    r.total_ms = static_cast<float>(timer.elapsed_ms());
    r.rmse = compute_rmse(pred, query.y);
    return r;
}

RunSummary run_cuda_naive(const DataSet& train, const DataSet& query, int k) {
    const auto gpu = knn_predict_cuda_naive(train.X, train.y, train.rows, train.cols, query.X, query.rows, k);

    RunSummary r;
    r.mode = "naive";
    r.total_ms = gpu.total_ms;
    r.h2d_ms = gpu.h2d_ms;
    r.kernel_ms = gpu.kernel_ms;
    r.d2h_ms = gpu.d2h_ms;
    r.post_ms = gpu.post_ms;
    r.rmse = compute_rmse(gpu.predictions, query.y);
    return r;
}

RunSummary run_cuda_optimized(const DataSet& train, const DataSet& query, int k) {
    const auto gpu = knn_predict_cuda_tiled(train.X, train.y, train.rows, train.cols, query.X, query.rows, k);

    RunSummary r;
    r.mode = "optimized";
    r.total_ms = gpu.total_ms;
    r.h2d_ms = gpu.h2d_ms;
    r.kernel_ms = gpu.kernel_ms;
    r.d2h_ms = gpu.d2h_ms;
    r.post_ms = gpu.post_ms;
    r.rmse = compute_rmse(gpu.predictions, query.y);
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parse_args(argc, argv);

        DataSet train;
        DataSet query;
        load_data(cfg, train, query);

        if (train.rows <= 0 || train.cols <= 0 || query.rows <= 0) {
            throw std::runtime_error("Loaded dataset is empty or invalid");
        }
        if (cfg.k <= 0) {
            throw std::runtime_error("k must be positive");
        }

        std::cout << "Dataset info:\n";
        std::cout << "  train_rows = " << train.rows << '\n';
        std::cout << "  query_rows = " << query.rows << '\n';
        std::cout << "  cols       = " << train.cols << '\n';
        std::cout << "  k          = " << cfg.k << '\n';
        std::cout << "  data_mode  = " << cfg.data << '\n';
        std::cout << "  run_mode   = " << cfg.mode << '\n';

        std::vector<RunSummary> results;

        if (cfg.mode == "cpu" || cfg.mode == "all") {
            results.push_back(run_cpu(train, query, cfg.k));
        }
        if (cfg.mode == "naive" || cfg.mode == "all") {
            results.push_back(run_cuda_naive(train, query, cfg.k));
        }
        if (cfg.mode == "optimized" || cfg.mode == "all") {
            results.push_back(run_cuda_optimized(train, query, cfg.k));
        }

        if (results.empty()) {
            throw std::runtime_error("No runs executed: invalid --mode");
        }

        for (const auto& r : results) {
            print_summary(r);
            if (!cfg.csv_out.empty()) {
                append_csv_row(cfg.csv_out, cfg, train.rows, query.rows, train.cols, r);
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nERROR: " << e.what() << '\n';
        return 1;
    }
}
