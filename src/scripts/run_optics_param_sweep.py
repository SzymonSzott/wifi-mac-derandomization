import os
import shlex
import subprocess
from itertools import product
from pathlib import Path

import pandas as pd


# =========================
# DATASET CONFIGURATION
# =========================

DATASET = "italian_without_xiaomi"
INPUT_TYPE = "pcap"
LABEL_FROM = "folder"
WORKERS = "8"

CSV_PATH = "outputs/experiments/3_optics_italian_without_xiaomi.csv"


# =========================
# BASE COMMAND
# =========================

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


def xi_to_name(xi):
    """
    Converts e.g. 0.05 to 0p05 so it can be used in the scenario name.
    """
    return str(xi).replace(".", "p")


def already_done(min_samples, xi, min_cluster_size):
    """
    Checks whether the given configuration is already in the CSV.
    This allows resuming the experiment after an interruption.
    """
    if not os.path.exists(CSV_PATH):
        return False

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        return False

    if df.empty:
        return False

    required_columns = {"min_samples", "xi", "min_cluster_size"}
    if not required_columns.issubset(df.columns):
        return False

    df["min_samples"] = pd.to_numeric(df["min_samples"], errors="coerce")
    df["xi"] = pd.to_numeric(df["xi"], errors="coerce")
    df["min_cluster_size"] = pd.to_numeric(df["min_cluster_size"], errors="coerce")

    mask = (
        (df["min_samples"] == min_samples) &
        (df["xi"].round(5) == round(xi, 5)) &
        (df["min_cluster_size"] == min_cluster_size)
    )

    return bool(mask.any())


def run_one(min_samples, xi, min_cluster_size, prefix):
    """
    Launches a single main.py run for a single OPTICS configuration.
    """

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
        f"{prefix}_"
        f"ms{min_samples}_"
        f"xi{xi_to_name(xi)}_"
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


def run_one_factor_tests():
    """
    Phase A:
    Vary one parameter at a time relative to the base configuration.
    This shows which parameter affects the result the most.
    """

    base_min_samples = 2
    base_xi = 0.05
    base_min_cluster_size = 15

    min_samples_values = [2, 3, 5, 8, 13, 21]
    xi_values = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
    min_cluster_size_values = [5, 10, 15, 20, 30, 50]

    print("\n\n### ONE-FACTOR TESTS: min_samples ###", flush=True)
    for ms in min_samples_values:
        run_one(
            min_samples=ms,
            xi=base_xi,
            min_cluster_size=base_min_cluster_size,
            prefix="onefactor_min_samples",
        )

    print("\n\n### ONE-FACTOR TESTS: xi ###", flush=True)
    for xi in xi_values:
        run_one(
            min_samples=base_min_samples,
            xi=xi,
            min_cluster_size=base_min_cluster_size,
            prefix="onefactor_xi",
        )

    print("\n\n### ONE-FACTOR TESTS: min_cluster_size ###", flush=True)
    for mcs in min_cluster_size_values:
        run_one(
            min_samples=base_min_samples,
            xi=base_xi,
            min_cluster_size=mcs,
            prefix="onefactor_mcs",
        )


def run_small_grid():
    """
    Phase B:
    Small grid around sensible values.
    This is the actual parameter stability analysis.
    """

    min_samples_values = [2, 3, 5, 8]
    xi_values = [0.03, 0.05, 0.08, 0.10]
    min_cluster_size_values = [10, 15, 20, 30]

    print("\n\n### SMALL GRID ###", flush=True)

    for ms, xi, mcs in product(
        min_samples_values,
        xi_values,
        min_cluster_size_values,
    ):
        run_one(
            min_samples=ms,
            xi=xi,
            min_cluster_size=mcs,
            prefix="grid",
        )


if __name__ == "__main__":
    Path("outputs/experiments").mkdir(parents=True, exist_ok=True)

    run_one_factor_tests()
    run_small_grid()

    print("\nDONE.", flush=True)
    print(f"Results CSV: {CSV_PATH}", flush=True)