# CUDA RentSense kNN

Небольшой учебный проект по CUDA: kNN-регрессия для оценки аренды по табличным признакам.

В проекте есть три режима:
- `cpu` — последовательный baseline на C++
- `naive` — CUDA kernel с расчётом попарных расстояний из global memory
- `optimized` — CUDA kernel с tiled shared memory

## Структура

```text
cuda-rentsense-knn/
├─ CMakeLists.txt
├─ include/
├─ src/
├─ data/
└─ scripts/
```

## Сборка

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Если `CUDA_ARCHITECTURES native` не подходит для вашей карты, задайте архитектуру вручную в `CMakeLists.txt`.

## Быстрый запуск

На синтетических данных:

```bash
./build/cuda_rentsense_knn --mode all --data synthetic --N 10000 --Q 512 --D 32 --k 5
```

С записью результатов в CSV:

```bash
./build/cuda_rentsense_knn \
  --mode all \
  --data synthetic \
  --N 20000 \
  --Q 1024 \
  --D 32 \
  --k 5 \
  --csv results/benchmark.csv
```

## Подготовка собственных данных

Скрипт `data/prepare_dataset.py` читает CSV, оставляет числовые признаки, делает split train/query, нормализацию и сохраняет бинарные файлы.

Пример:

```bash
python3 data/prepare_dataset.py \
  --csv your_data.csv \
  --target price \
  --out_dir data/processed
```

После этого можно запускать бинарный режим:

```bash
./build/cuda_rentsense_knn \
  --mode all \
  --data binary \
  --train_features data/processed/X_train.bin \
  --train_labels data/processed/y_train.bin \
  --query_features data/processed/X_query.bin \
  --query_labels data/processed/y_query.bin
```

## Профилирование

Nsight Compute:

```bash
bash scripts/profile_ncu.sh
```

Nsight Systems:

```bash
bash scripts/profile_nsys.sh
```

## Что измеряется

- `total_ms`
- `h2d_ms`
- `kernel_ms`
- `d2h_ms`
- `post_ms`
- `rmse`

## Идея оптимизации

Основная вычислительная часть — матрица попарных расстояний между `query` и `train` объектами. Эта операция естественно распараллеливается. В optimized-версии часть признаков подгружается в shared memory, чтобы уменьшить число чтений из global memory.
