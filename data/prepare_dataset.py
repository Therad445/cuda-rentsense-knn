#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import struct

import numpy as np
import pandas as pd


def write_matrix(path: str, matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype=np.float32, order="C")
    rows, cols = matrix.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", rows, cols))
        f.write(matrix.tobytes(order="C"))


def write_vector(path: str, vector: np.ndarray) -> None:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    rows = vector.shape[0]
    with open(path, "wb") as f:
        f.write(struct.pack("<i", rows))
        f.write(vector.tobytes(order="C"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to source CSV")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--out_dir", default="data/processed", help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found")

    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if args.target not in numeric_df.columns:
        raise ValueError(
            f"Target column '{args.target}' is not numeric or was dropped by select_dtypes"
        )

    y = numeric_df.pop(args.target).to_numpy(dtype=np.float32)
    X = numeric_df.to_numpy(dtype=np.float32)

    if X.shape[0] < 10:
        raise ValueError("Too few rows after preprocessing")

    rng = np.random.default_rng(args.seed)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)

    X = X[indices]
    y = y[indices]

    split = int(X.shape[0] * args.train_ratio)
    split = max(1, min(split, X.shape[0] - 1))

    X_train = X[:split]
    y_train = y[:split]
    X_query = X[split:]
    y_query = y[split:]

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0

    X_train = (X_train - mean) / std
    X_query = (X_query - mean) / std

    os.makedirs(args.out_dir, exist_ok=True)

    write_matrix(os.path.join(args.out_dir, "X_train.bin"), X_train)
    write_vector(os.path.join(args.out_dir, "y_train.bin"), y_train)
    write_matrix(os.path.join(args.out_dir, "X_query.bin"), X_query)
    write_vector(os.path.join(args.out_dir, "y_query.bin"), y_query)

    print("Saved:")
    print(f"  {args.out_dir}/X_train.bin  shape={X_train.shape}")
    print(f"  {args.out_dir}/y_train.bin  shape={y_train.shape}")
    print(f"  {args.out_dir}/X_query.bin  shape={X_query.shape}")
    print(f"  {args.out_dir}/y_query.bin  shape={y_query.shape}")


if __name__ == "__main__":
    main()
