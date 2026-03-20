#!/usr/bin/env bash
set -euo pipefail

mkdir -p build results

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

OUT="results/benchmark.csv"
rm -f "${OUT}"

for N in 5000 10000 20000 50000; do
  ./build/cuda_rentsense_knn \
    --mode all \
    --data synthetic \
    --N "${N}" \
    --Q 512 \
    --D 32 \
    --k 5 \
    --csv "${OUT}"
done

echo "Benchmark saved to ${OUT}"
