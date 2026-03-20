#pragma once

#include <vector>

std::vector<float> knn_predict_cpu(const std::vector<float>& X_train,
                                   const std::vector<float>& y_train,
                                   int train_rows,
                                   int cols,
                                   const std::vector<float>& X_query,
                                   int query_rows,
                                   int k);

std::vector<float> knn_predict_from_distances_cpu(const std::vector<float>& dist,
                                                  const std::vector<float>& y_train,
                                                  int query_rows,
                                                  int train_rows,
                                                  int k);
