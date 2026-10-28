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

CSV_PATH = "outputs/experiments/3_optics_xi_wide.csv"

BASE_CMD = [
    "python", "-m", "src.cli.main",
    "-d", DATASET,
    "-i", INPUT_TYPE,
    "-l",
    "--label-from", LABEL_FROM,
    "--workers", WORKERS,
    "--clustering-method", "optics",
    "--reverted-table",
    "--experiments-csv", CSV_PATH,
]


def value_to_name(value):
    return str(value).replace(".", "p")


def already_done(min_samples, xi, min_cluster_size):
    if not os.path.exists(CSV_PATH):
        return False

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        return False

    if df.empty:
        return False

    required = {"min_samples", "xi", "min_cluster_size"}
    if not required.issubset(df.columns):
        return False

    df["min_samples"] = pd.to_numeric(df["min_samples"], errors="coerce")
    df["xi"] = pd.to_numeric(df["xi"], errors="coerce")
    df["min_cluster_size"] = pd.to_numeric(df["min_cluster_size"], errors="coerce")

    mask = (
        (df["min_samples"] == min_samples) &
        (df["xi"].round(8) == round(xi, 8)) &
        (df["min_cluster_size"] == min_cluster_size)
    )

    return bool(mask.any())


def run_one(min_samples, xi, min_cluster_size):
    if already_done(min_samples, xi, min_cluster_size):
        print(
            "\nSKIP:",
            f"min_samples={min_samples},",
            f"xi={xi},",
            f"min_cluster_size={min_cluster_size}",
            flush=True,
        )
        return

    scenario = (
        f"xiwide_"
        f"ms{min_samples}_"
        f"xi{value_to_name(xi)}_"
        f"mcs{min_cluster_size}"
    )

    cmd = BASE_CMD + [
        "-s", scenario,
        "--min-samples", str(min_samples),
        "--xi", str(xi),
        "--min-cluster-size", str(min_cluster_size),
    ]

    print("\n" + "=" * 100, flush=True)
    print("RUN:", scenario, flush=True)
    print("COMMAND:", flush=True)
    print(shlex.join(cmd), flush=True)
    print("=" * 100, flush=True)

    subprocess.run(cmd, check=True)


def main():
    Path("outputs/experiments").mkdir(parents=True, exist_ok=True)

    min_samples_values = [2, 3, 5, 8, 13, 21]
    xi_values = [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70]
    min_cluster_size_values = [15]

    for min_samples, xi, min_cluster_size in product(
        min_samples_values,
        xi_values,
        min_cluster_size_values,
    ):
        run_one(
            min_samples=min_samples,
            xi=xi,
            min_cluster_size=min_cluster_size,
        )

    print("\nDONE.", flush=True)
    print(f"Results CSV: {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()