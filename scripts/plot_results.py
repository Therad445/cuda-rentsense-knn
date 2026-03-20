#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/benchmark.csv")
    parser.add_argument("--out_dir", default="results/figures")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.out_dir, exist_ok=True)

    df = df.sort_values(["mode", "N"])

    plt.figure(figsize=(8, 5))
    for mode in df["mode"].unique():
        sub = df[df["mode"] == mode]
        plt.plot(sub["N"], sub["total_ms"], marker="o", label=mode)
    plt.xlabel("Train rows (N)")
    plt.ylabel("Total time, ms")
    plt.title("Total execution time vs dataset size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "time_vs_n.png"), dpi=180)
    plt.close()

    cpu = df[df["mode"] == "cpu"][["N", "total_ms"]].rename(columns={"total_ms": "cpu_ms"})
    merged = df.merge(cpu, on="N", how="left")
    merged["speedup_vs_cpu"] = merged["cpu_ms"] / merged["total_ms"]

    plt.figure(figsize=(8, 5))
    for mode in ["naive", "optimized"]:
        sub = merged[merged["mode"] == mode]
        if not sub.empty:
            plt.plot(sub["N"], sub["speedup_vs_cpu"], marker="o", label=mode)
    plt.xlabel("Train rows (N)")
    plt.ylabel("Speedup vs CPU")
    plt.title("GPU speedup vs CPU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "speedup_vs_cpu.png"), dpi=180)
    plt.close()

    latest_n = df["N"].max()
    sub = df[df["N"] == latest_n].copy()

    plt.figure(figsize=(8, 5))
    x = range(len(sub))
    plt.bar(x, sub["h2d_ms"], label="H2D")
    plt.bar(x, sub["kernel_ms"], bottom=sub["h2d_ms"], label="Kernel")
    plt.bar(x, sub["d2h_ms"], bottom=sub["h2d_ms"] + sub["kernel_ms"], label="D2H")
    plt.bar(
        x,
        sub["post_ms"],
        bottom=sub["h2d_ms"] + sub["kernel_ms"] + sub["d2h_ms"],
        label="Post",
    )
    plt.xticks(list(x), sub["mode"])
    plt.ylabel("Time, ms")
    plt.title(f"Time breakdown for N={latest_n}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "breakdown_latest_n.png"), dpi=180)
    plt.close()

    print(f"Saved figures to {args.out_dir}")


if __name__ == "__main__":
    main()
