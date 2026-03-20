#!/usr/bin/env bash
set -euo pipefail

mkdir -p build results/nsys

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

nsys profile -o results/nsys/optimized_timeline \
  ./build/cuda_rentsense_knn \
  --mode optimized \
  --data synthetic \
  --N 20000 \
  --Q 512 \
  --D 32 \
  --k 5

echo "Nsight Systems report saved in results/nsys/"
