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

CSV_PATH = "outputs/experiments/4_dbscan_italian_without_xiaomi.csv"

BASE_CMD = [
    "python", "-m", "src.cli.main",
    "-d", DATASET,
    "-i", INPUT_TYPE,
    "-l",
    "--label-from", LABEL_FROM,
    "--workers", WORKERS,
    "--clustering-method", "dbscan",
    "--reverted-table",
    "--experiments-csv", CSV_PATH,
]


def value_to_name(value):
    return str(value).replace(".", "p")


def already_done(eps, min_samples):
    if not os.path.exists(CSV_PATH):
        return False

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        return False

    if df.empty:
        return False

    required = {"eps", "min_samples"}
    if not required.issubset(df.columns):
        return False

    df["eps"] = pd.to_numeric(df["eps"], errors="coerce")
    df["min_samples"] = pd.to_numeric(df["min_samples"], errors="coerce")

    mask = (
        (df["eps"].round(8) == round(eps, 8)) &
        (df["min_samples"] == min_samples)
    )

    return bool(mask.any())


def run_one(eps, min_samples, prefix):
    if already_done(eps, min_samples):
        print(
            "\nSKIP:",
            f"eps={eps},",
            f"min_samples={min_samples}",
            flush=True,
        )
        return

    scenario = (
        f"{prefix}_"
        f"eps{value_to_name(eps)}_"
        f"ms{min_samples}"
    )

    cmd = BASE_CMD + [
        "-s", scenario,
        "--eps", str(eps),
        "--min-samples", str(min_samples),
    ]

    print("\n" + "=" * 100, flush=True)
    print("RUN:", scenario, flush=True)
    print("COMMAND:", flush=True)
    print(shlex.join(cmd), flush=True)
    print("=" * 100, flush=True)

    subprocess.run(cmd, check=True)


def run_one_factor_tests():
    base_eps = 0.5
    base_min_samples = 5

    eps_values = [
        0.01, 0.02, 0.03, 0.05, 0.08,
        0.10, 0.15, 0.20, 0.30, 0.50,
        0.75, 1.00, 1.50, 2.00, 3.00, 5.00,
    ]

    min_samples_values = [2, 3, 5, 8, 13, 21]

    print("\n\n### ONE-FACTOR TESTS: eps ###", flush=True)
    for eps in eps_values:
        run_one(
            eps=eps,
            min_samples=base_min_samples,
            prefix="onefactor_eps",
        )

    print("\n\n### ONE-FACTOR TESTS: min_samples ###", flush=True)
    for min_samples in min_samples_values:
        run_one(
            eps=base_eps,
            min_samples=min_samples,
            prefix="onefactor_ms",
        )


def run_grid():
    eps_values = [
    0.01, 0.02, 0.03, 0.05, 0.08,
    0.10, 0.15, 0.20, 0.30, 0.50,
    0.75, 1.00, 1.50, 2.00, 3.00, 5.00,
    7.50, 10.00, 15.00, 20.00, 30.00, 50.00, 100.00,
    ]

    min_samples_values = [2, 3, 5, 8, 13, 21]

    print("\n\n### DBSCAN GRID ###", flush=True)

    for eps, min_samples in product(eps_values, min_samples_values):
        run_one(
            eps=eps,
            min_samples=min_samples,
            prefix="grid",
        )


if __name__ == "__main__":
    Path("outputs/experiments").mkdir(parents=True, exist_ok=True)

    run_one_factor_tests()
    run_grid()

    print("\nDONE.", flush=True)
    print(f"Results CSV: {CSV_PATH}", flush=True)