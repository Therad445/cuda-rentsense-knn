#!/usr/bin/env bash
set -euo pipefail

mkdir -p build results/ncu

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

ncu -o results/ncu/naive_profile \
  ./build/cuda_rentsense_knn \
  --mode naive \
  --data synthetic \
  --N 20000 \
  --Q 512 \
  --D 32 \
  --k 5

ncu -o results/ncu/optimized_profile \
  ./build/cuda_rentsense_knn \
  --mode optimized \
  --data synthetic \
  --N 20000 \
  --Q 512 \
  --D 32 \
  --k 5

echo "Nsight Compute reports saved in results/ncu/"
