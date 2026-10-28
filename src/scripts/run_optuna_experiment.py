import argparse
import os
import shlex
import subprocess
from pathlib import Path

import optuna
import pandas as pd


def value_to_name(value):
    return str(value).replace(".", "p")


def build_score(row):
    """
    Helper score for Optuna.
    You can change the weights if you consider, e.g., DCMS more important than V-measure.
    """

    v_measure = float(row.get("v_measure", 0.0))
    dcms_weighted = float(row.get("dcms_weighted", 0.0))
    devices_one_cluster_pct = float(row.get("devices_one_cluster_pct", 0.0))
    pure_clusters_pct = float(row.get("pure_clusters_pct", 0.0))
    noise_pct = float(row.get("noise_pct", 100.0))

    score = (
        0.40 * v_measure +
        0.25 * dcms_weighted +
        0.20 * (devices_one_cluster_pct / 100.0) +
        0.15 * (pure_clusters_pct / 100.0) -
        0.10 * (noise_pct / 100.0)
    )

    return score


def read_result_for_scenario(csv_path, scenario):
    """
    Reads the CSV and returns the last row for the given scenario.
    """

    if not os.path.exists(csv_path):
        raise RuntimeError(f"CSV does not exist yet: {csv_path}")

    df = pd.read_csv(csv_path)

    if "scenario" not in df.columns:
        raise RuntimeError("CSV does not contain 'scenario' column")

    matched = df[df["scenario"] == scenario]

    if matched.empty:
        raise RuntimeError(f"No row found in CSV for scenario: {scenario}")

    return matched.iloc[-1]


def run_main_command(cmd):
    print("\n" + "=" * 100, flush=True)
    print("COMMAND:", flush=True)
    print(shlex.join(cmd), flush=True)
    print("=" * 100, flush=True)

    subprocess.run(cmd, check=True)


def make_base_cmd(args):
    return [
        "python", "-m", "src.cli.main",
        "-d", args.dataset,
        "-i", args.input_type,
        "-l",
        "--label-from", args.label_from,
        "--workers", str(args.workers),
        "--clustering-method", args.method,
        "--reverted-table",
        "--experiments-csv", args.csv_path,
    ]


def suggest_optics_params(trial):
    return {
        "min_samples": trial.suggest_int("min_samples", 2, 21),
        "xi": trial.suggest_float("xi", 0.01, 0.20),
        "min_cluster_size": trial.suggest_int("min_cluster_size", 5, 50),
    }


def suggest_dbscan_params(trial):
    return {
        "eps": trial.suggest_float("eps", 0.05, 5.0, log=True),
        "min_samples": trial.suggest_int("min_samples", 2, 21),
    }


def suggest_hdbscan_params(trial):
    return {
        "min_cluster_size": trial.suggest_int("min_cluster_size", 5, 50),
        "min_samples": trial.suggest_int("min_samples", 2, 21),
        "cluster_selection_epsilon": trial.suggest_float(
            "cluster_selection_epsilon",
            0.0,
            2.0,
        ),
    }


def build_scenario(method, trial_number, params):
    parts = [f"optuna_{method}", f"trial{trial_number:04d}"]

    for key, value in params.items():
        parts.append(f"{key}{value_to_name(value)}")

    return "_".join(parts)


def build_method_args(method, params):
    if method == "optics":
        return [
            "--min-samples", str(params["min_samples"]),
            "--xi", str(params["xi"]),
            "--min-cluster-size", str(params["min_cluster_size"]),
        ]

    if method == "dbscan":
        return [
            "--eps", str(params["eps"]),
            "--min-samples", str(params["min_samples"]),
        ]

    if method == "hdbscan":
        return [
            "--min-cluster-size", str(params["min_cluster_size"]),
            "--min-samples", str(params["min_samples"]),
            "--cluster-selection-epsilon", str(params["cluster_selection_epsilon"]),
        ]

    raise ValueError(f"Unknown method: {method}")


def make_objective(args):
    base_cmd = make_base_cmd(args)

    def objective(trial):
        if args.method == "optics":
            params = suggest_optics_params(trial)
        elif args.method == "dbscan":
            params = suggest_dbscan_params(trial)
        elif args.method == "hdbscan":
            params = suggest_hdbscan_params(trial)
        else:
            raise ValueError(f"Unknown method: {args.method}")

        scenario = build_scenario(args.method, trial.number, params)

        cmd = base_cmd + [
            "-s", scenario,
            *build_method_args(args.method, params),
        ]

        run_main_command(cmd)

        row = read_result_for_scenario(args.csv_path, scenario)
        score = build_score(row)

        trial.set_user_attr("scenario", scenario)
        trial.set_user_attr("v_measure", float(row.get("v_measure", 0.0)))
        trial.set_user_attr("dcms_weighted", float(row.get("dcms_weighted", 0.0)))
        trial.set_user_attr("noise_pct", float(row.get("noise_pct", 100.0)))
        trial.set_user_attr(
            "devices_one_cluster_pct",
            float(row.get("devices_one_cluster_pct", 0.0)),
        )
        trial.set_user_attr("report_file", row.get("report_file", ""))

        print("\nRESULT:", flush=True)
        print(f"scenario={scenario}", flush=True)
        print(f"score={score:.6f}", flush=True)
        print(f"v_measure={row.get('v_measure')}", flush=True)
        print(f"dcms_weighted={row.get('dcms_weighted')}", flush=True)
        print(f"noise_pct={row.get('noise_pct')}", flush=True)
        print(f"devices_one_cluster_pct={row.get('devices_one_cluster_pct')}", flush=True)

        return score

    return objective


def save_trials_summary(study, out_path):
    rows = []

    for trial in study.trials:
        row = {
            "trial_number": trial.number,
            "value": trial.value,
            "state": str(trial.state),
        }

        row.update(trial.params)
        row.update(trial.user_attrs)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        choices=["optics", "dbscan", "hdbscan"],
        required=True,
    )

    parser.add_argument("--dataset", default="italian_without_xiaomi")
    parser.add_argument("--input-type", default="pcap")
    parser.add_argument("--label-from", default="folder")
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument("--n-trials", type=int, default=50)

    parser.add_argument(
        "--csv-path",
        default=None,
        help="CSV where main.py appends results. If not provided, a method-specific path is used.",
    )

    parser.add_argument(
        "--study-db",
        default=None,
        help="SQLite file for Optuna study. Allows resuming.",
    )

    parser.add_argument(
        "--study-name",
        default=None,
        help="Optuna study name. Allows resuming.",
    )

    args = parser.parse_args()

    Path("outputs/experiments").mkdir(parents=True, exist_ok=True)

    if args.csv_path is None:
        args.csv_path = f"outputs/experiments/optuna_{args.dataset}_{args.method}.csv"

    if args.study_db is None:
        args.study_db = f"outputs/experiments/optuna_{args.dataset}_{args.method}.db"

    if args.study_name is None:
        args.study_name = f"optuna_{args.dataset}_{args.method}"

    storage = f"sqlite:///{args.study_db}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )

    study.optimize(
        make_objective(args),
        n_trials=args.n_trials,
    )

    summary_path = f"outputs/experiments/optuna_{args.dataset}_{args.method}_trials.csv"
    save_trials_summary(study, summary_path)

    print("\n" + "=" * 100)
    print("OPTUNA DONE")
    print("=" * 100)

    print("\nBest trial:")
    print("number:", study.best_trial.number)
    print("score:", study.best_trial.value)
    print("params:", study.best_trial.params)
    print("user_attrs:", study.best_trial.user_attrs)

    print("\nFiles:")
    print("results csv:", args.csv_path)
    print("study db:", args.study_db)
    print("summary csv:", summary_path)


if __name__ == "__main__":
    main()