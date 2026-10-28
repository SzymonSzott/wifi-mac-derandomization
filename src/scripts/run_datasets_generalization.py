import os
import shlex
import subprocess
from pathlib import Path

import pandas as pd


# =========================
# CONFIG
# =========================

DATASETS = [
    {"name": "italian_without_xiaomi", "input_type": "pcap", "label_from": "folder"},
    {"name": "italian", "input_type": "pcap", "label_from": "folder"},
    {"name": "awid3_pr_only", "input_type": "pcap", "label_from": "file"},
    

]

WORKERS = "8"

OUT_DIR = "outputs/experiments/datasets_generalization"


# =========================
# SELECTED CONFIGURATIONS
# =========================

EXPERIMENTS = [
    {
        "id": "OPTICS_A",
        "method": "optics",
        "params": {
            "min_samples": 13,
            "xi": 0.05,
            "min_cluster_size": 15,
        },
    },
    {
        "id": "OPTICS_B",
        "method": "optics",
        "params": {
            "min_samples": 5,
            "xi": 0.05,
            "min_cluster_size": 15,
        },
    },
    {
        "id": "OPTICS_C",
        "method": "optics",
        "params": {
            "min_samples": 8,
            "xi": 0.05,
            "min_cluster_size": 15,
        },
    },
    {
        "id": "DBSCAN_A",
        "method": "dbscan",
        "params": {
            "eps": 0.50,
            "min_samples": 13,
        },
    },
    {
        "id": "DBSCAN_B",
        "method": "dbscan",
        "params": {
            "eps": 1.50,
            "min_samples": 13,
        },
    },
    {
        "id": "DBSCAN_C",
        "method": "dbscan",
        "params": {
            "eps": 0.50,
            "min_samples": 5,
        },
    },
    {
        "id": "HDBSCAN_A",
        "method": "hdbscan",
        "params": {
            "min_cluster_size": 5,
            "min_samples": 13,
            "cluster_selection_epsilon": 0.0,
        },
    },
    {
        "id": "HDBSCAN_B",
        "method": "hdbscan",
        "params": {
            "min_cluster_size": 10,
            "min_samples": 13,
            "cluster_selection_epsilon": 0.0,
        },
    },
    {
        "id": "HDBSCAN_C",
        "method": "hdbscan",
        "params": {
            "min_cluster_size": 15,
            "min_samples": 13,
            "cluster_selection_epsilon": 0.0,
        },
    },
]


# =========================
# HELPERS
# =========================

def value_to_name(value):
    return str(value).replace(".", "p")


def param_to_cli_arg(name):
    return "--" + name.replace("_", "-")


def already_done_by_scenario(csv_path, scenario):
    if not os.path.exists(csv_path):
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False

    if df.empty or "scenario" not in df.columns:
        return False

    return bool((df["scenario"].astype(str) == scenario).any())


def build_scenario(dataset_name, experiment):
    method = experiment["method"]
    exp_id = experiment["id"]
    params = experiment["params"]

    parts = [dataset_name, exp_id]

    if method == "optics":
        parts.append(f"ms{params['min_samples']}")
        parts.append(f"xi{value_to_name(params['xi'])}")
        parts.append(f"mcs{params['min_cluster_size']}")

    elif method == "dbscan":
        parts.append(f"eps{value_to_name(params['eps'])}")
        parts.append(f"ms{params['min_samples']}")

    elif method == "hdbscan":
        parts.append(f"mcs{params['min_cluster_size']}")
        parts.append(f"ms{params['min_samples']}")
        parts.append(f"cse{value_to_name(params['cluster_selection_epsilon'])}")

    return "_".join(parts)


def build_command(dataset_name, input_type, label_from, experiment, csv_path):
    method = experiment["method"]
    scenario = build_scenario(dataset_name, experiment)

    cmd = [
        "python", "-m", "src.cli.main",
        "-d", dataset_name,
        "-i", input_type,
        "-l",
        "--label-from", label_from,
        "--workers", WORKERS,
        "--clustering-method", method,
        "--reverted-table",
        "-s", scenario,
        "--experiments-csv", csv_path,
    ]

    for name, value in experiment["params"].items():
        cmd.extend([param_to_cli_arg(name), str(value)])

    return cmd


def run_one(dataset_config, experiment):
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    dataset_name = dataset_config["name"]
    input_type = dataset_config["input_type"]
    label_from = dataset_config["label_from"]

    csv_path = os.path.join(
        OUT_DIR,
        f"generalization_{dataset_name}.csv",
    )

    scenario = build_scenario(dataset_name, experiment)

    if already_done_by_scenario(csv_path, scenario):
        print("\nSKIP:", scenario, flush=True)
        return

    cmd = build_command(
        dataset_name=dataset_name,
        input_type=input_type,
        label_from=label_from,
        experiment=experiment,
        csv_path=csv_path,
    )

    print("\n" + "=" * 100, flush=True)
    print("RUN:", scenario, flush=True)
    print("DATASET:", dataset_name, flush=True)
    print("INPUT TYPE:", input_type, flush=True)
    print("LABEL FROM:", label_from, flush=True)
    print("METHOD:", experiment["method"], flush=True)
    print("COMMAND:", flush=True)
    print(shlex.join(cmd), flush=True)
    print("=" * 100, flush=True)

    subprocess.run(cmd, check=True)


def main():
    if not DATASETS:
        raise ValueError(
            "DATASETS is empty. Add dataset configs to the DATASETS list."
        )

    total = len(DATASETS) * len(EXPERIMENTS)

    print("\n### DATASET GENERALIZATION EXPERIMENTS ###", flush=True)
    print(f"Datasets: {DATASETS}", flush=True)
    print(f"Configurations per dataset: {len(EXPERIMENTS)}", flush=True)
    print(f"Expected runs: {total}", flush=True)
    print(f"Output directory: {OUT_DIR}", flush=True)

    for dataset_config in DATASETS:
        print("\n" + "#" * 100, flush=True)
        print(f"DATASET: {dataset_config['name']}", flush=True)
        print(f"INPUT TYPE: {dataset_config['input_type']}", flush=True)
        print(f"LABEL FROM: {dataset_config['label_from']}", flush=True)
        print("#" * 100, flush=True)

        for experiment in EXPERIMENTS:
            run_one(dataset_config, experiment)

    print("\nDONE.", flush=True)
    print(f"Results saved under: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()