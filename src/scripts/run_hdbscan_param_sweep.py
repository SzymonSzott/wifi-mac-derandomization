import os
import shlex
import subprocess
from itertools import product
from pathlib import Path

import pandas as pd


DATASET = "italian_without_xiaomi"
INPUT_TYPE = "pcap"
LABEL_FROM = "folder"
WORKERS = "8"

CSV_PATH = "outputs/experiments/5_hdbscan_italian_without_xiaomi.csv"

BASE_CMD = [
    "python", "-m", "src.cli.main",
    "-d", DATASET,
    "-i", INPUT_TYPE,
    "-l",
    "--label-from", LABEL_FROM,
    "--workers", WORKERS,
    "--clustering-method", "hdbscan",
    "--reverted-table",
    "--experiments-csv", CSV_PATH,
]


def value_to_name(value):
    return str(value).replace(".", "p")


def already_done(min_cluster_size, min_samples, cluster_selection_epsilon):
    if not os.path.exists(CSV_PATH):
        return False

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        return False

    if df.empty:
        return False

    required = {"min_cluster_size", "min_samples", "cluster_selection_epsilon"}
    if not required.issubset(df.columns):
        return False

    df["min_cluster_size"] = pd.to_numeric(
        df["min_cluster_size"],
        errors="coerce",
    )

    df["min_samples"] = pd.to_numeric(
        df["min_samples"],
        errors="coerce",
    )

    df["cluster_selection_epsilon"] = pd.to_numeric(
        df["cluster_selection_epsilon"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

    mask = (
        (df["min_cluster_size"] == min_cluster_size) &
        (df["min_samples"] == min_samples) &
        (df["cluster_selection_epsilon"].round(8) == round(cluster_selection_epsilon, 8))
    )

    return bool(mask.any())


def run_one(min_cluster_size, min_samples, cluster_selection_epsilon):
    if already_done(min_cluster_size, min_samples, cluster_selection_epsilon):
        print(
            "\nSKIP:",
            f"min_cluster_size={min_cluster_size},",
            f"min_samples={min_samples},",
            f"cluster_selection_epsilon={cluster_selection_epsilon}",
            flush=True,
        )
        return

    scenario = (
        f"grid_"
        f"mcs{min_cluster_size}_"
        f"ms{min_samples}_"
        f"cse{value_to_name(cluster_selection_epsilon)}"
    )

    cmd = BASE_CMD + [
        "-s", scenario,
        "--min-cluster-size", str(min_cluster_size),
        "--min-samples", str(min_samples),
        "--cluster-selection-epsilon", str(cluster_selection_epsilon),
    ]

    print("\n" + "=" * 100, flush=True)
    print("RUN:", scenario, flush=True)
    print("COMMAND:", flush=True)
    print(shlex.join(cmd), flush=True)
    print("=" * 100, flush=True)

    subprocess.run(cmd, check=True)


def main():
    Path("outputs/experiments").mkdir(parents=True, exist_ok=True)

    min_cluster_size_values = [5, 10, 15, 30, 50, 100]
    min_samples_values = [2, 3, 5, 8, 13, 21]
    cluster_selection_epsilon_values = [0.0, 0.05, 0.20, 1.00]

    total = (
        len(min_cluster_size_values)
        * len(min_samples_values)
        * len(cluster_selection_epsilon_values)
    )

    print("\n### HDBSCAN PARAMETER SWEEP ###", flush=True)
    print(f"Expected unique configurations: {total}", flush=True)
    print(f"Results CSV: {CSV_PATH}", flush=True)

    for min_cluster_size, min_samples, cluster_selection_epsilon in product(
        min_cluster_size_values,
        min_samples_values,
        cluster_selection_epsilon_values,
    ):
        run_one(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
        )

    print("\nDONE.", flush=True)
    print(f"Results CSV: {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()