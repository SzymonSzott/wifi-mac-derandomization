from contextlib import redirect_stdout
import os
import argparse
from src.evaluation.validation import display_report, print_v_measure_report_simple, validation_python, display_reverted_table
from src.evaluation.experiment_csv import append_report_to_csv
from src.data_loaders.load_pcap import load_pcap
from src.data_loaders.load_json import load_json
from src.data_processing.feature_extraction import exctract_fields_values
from src.data_processing.calculate_distance import calculate_distance_matrix
from src.common.common import extract_mac_to_label
import numpy as np
from datetime import datetime


# random weights for distance calculation
rng = np.random.default_rng(45)  # for reproducibility

def rand(a = -1000,b = 1000):
    return rng.uniform(a,b)

# entropy-based weights
multiplier = 1000.0

def main():
    BASE_INPUT_DIR = os.path.normpath(
        os.path.join(os.path.dirname(__file__),"../../dataset")
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", "-d",  default="", help="Subdirectory within dataset")
    parser.add_argument("--input-type", "-i", choices=["json", "pcap"], default="json")
    parser.add_argument("--labeled", "-l", action="store_true")
    parser.add_argument("--label-from", choices=["folder", "file"], default="folder")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--scenario", "-s", type=str, default="default", help="Scenario name for report naming")
    parser.add_argument("--clustering-method", choices=["optics", "dbscan", "hdbscan"], default="optics")
    parser.add_argument("--reverted-table", action="store_true", help="Display reverted table (clusters -> labels)")
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--xi", type=float, default=0.05)
    parser.add_argument("--min-cluster-size", type=int, default=15)
    parser.add_argument("--max-eps", type=float, default=float("inf"))
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--cluster-selection-epsilon", type=float, default=0.0)
    parser.add_argument("--experiments-csv", type=str, default="outputs/experiments/experiment.csv", help="CSV file where one row per run will be appended")
    
    args = parser.parse_args()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    INPUT_DIR = os.path.normpath(os.path.join(BASE_INPUT_DIR, args.input_dir))
    
    print("Using directory:", INPUT_DIR)

    if args.clustering_method == "optics":
        from src.clustering.optics import optics_clustering as clustering, plot_optics_reachability as plot_func, extract_clusters
    elif args.clustering_method == "dbscan":
        from src.clustering.dbscan import dbscan_clustering as clustering, extract_clusters
        plot_func = None
    else:  # hdbscan
        from src.clustering.hdbscan import hdbscan_clustering as clustering, extract_clusters
        plot_func = None

    if args.input_type == "json":
        df = load_json(INPUT_DIR)
    else:
        df = load_pcap(
            INPUT_DIR,
            labeled=args.labeled,
            label_from=args.label_from,
            workers=args.workers,
    )


    fields_cfg = {
    # --- combined time and RSSI (this is a "summed" distance, see below) ---
    # This field is virtual: doesn't need to exist within the ingested data, becasue distance is calculated from RSSI and TIME together
    "RSSI_TIME": {
        "kind": "rssi_time",
        # dist_rssi_time(rssi1, time1, rssi2, time2, weight1, weight2, weight3, weight4)
        "weight1": 1.055,   # the base of the exponent for the RSSI difference (weight1 ** abs(delta_rssi))
        "weight2": 5.0,   # penalty/reward for adjustment in time
        "weight3": -5.0,   # offset
        "weight4": 0.1,   # time threshold (in seconds) for decision on being close/far
    },

    # --- DATA_RTS: list of rates ---
    # If ext() returns a list of numbers in string format, it's better to cast it to int before dist_data_rates
    "DATA_RTS_SUPP": {"kind": "data_rates", "weight": 25.0},
    "DATA_RTS_EXT":  {"kind": "data_rates", "weight": 25.0},

    # --- CAPabilities: hex strings -> bit_dist ---
    "HT_CAP":  {"kind": "hex_bit", "weight": 400, "penalty": 0.0},
    "HT_CAP_INFO": {"kind": "hex_bit", "weight": 400, "penalty": 0.0},
    "HT_AMPDU_PARMS": {"kind": "hex_bit", "weight": 400, "penalty": 0.0},
    "HT_MCS_SET": {"kind": "hex_bit", "weight": 400, "penalty": 0.0},
    "HT_EXT_CAP": {"kind": "hex_bit", "weight": 400, "penalty": 0.0},
    "HT_Tx": {"kind": "hex_bit", "weight": 400, "penalty": 0.0},
    "HT_ANTENNA": {"kind": "hex_bit", "weight": 400, "penalty": 0.0},
    "VHT_CAP": {"kind": "hex_bit", "weight": 600, "penalty": 0.0},
    "EXT_CAP": {"kind": "hex_bit", "weight": 100, "penalty": 0.0},

    # --- TAG_LIST: list of tags (strings) ---
    "TAG_LIST": {"kind": "ie_list", "weight": 15, "penalty": 1.0},

    # --- dicts ---
    # EXT_TAG: dist_ext_tag(dict1, dict2, weight1, weight2, weight3)
    "EXT_TAG": {
        "kind": "ext_tag",
        "weight1": 6.0,   # penalty for differing tags (diff_tags)
        "weight2": 100.0,   # weight bit_dist inside shared tags
        "weight3": 8.0,   # "bonus" for shared tags (it's subtracted here)
    },

    # VENDOR_SPEC: dist_vendor_spec(dict1, dict2, weight1, weight2, weight3)
    "VENDOR_SPEC": {
        "kind": "vendor_spec",
        "weight1": 6.0,   # penalty for different vendor OUIs
        "weight2": 100.0,   # weight bit_dist for shared dtype's
        "weight3": 8.0,   # reward for shared dtype's
    },
    "SSID": {"kind": "ssid", "weight": 25.0},
}


    fields_by_mac = exctract_fields_values(
        df,
        [
            "DATA_RTS_SUPP",
            "DATA_RTS_EXT",
            "HT_CAP",
            "VHT_CAP",
            "EXT_CAP",
            #"INTWORK",
            "VENDOR_SPEC",
            "EXT_TAG",
            "RSSI_TIME",
            "SSID",
            "TAG_LIST",
        ],
    )

    print("\nCalculating distance matrix...")
    distance_matrix = calculate_distance_matrix(fields_by_mac, fields_cfg, null_token="NULL")

    # Perform clustering
    method_name = args.clustering_method.upper()
    print(f"\nPerforming {method_name} clustering...")
    if args.clustering_method == "optics":
        model, labels = clustering(
            distance_matrix,
            min_samples=args.min_samples,
            xi=args.xi,
            min_cluster_size=args.min_cluster_size,
            max_eps=args.max_eps,
        )

    elif args.clustering_method == "dbscan":
        model, labels = clustering(
            distance_matrix,
            eps=args.eps,
            min_samples=args.min_samples,
        )

    elif args.clustering_method == "hdbscan":
        model, labels = clustering(
            distance_matrix,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
        )
    
    if plot_func:
        os.makedirs("outputs/plots", exist_ok=True)
        reachability_plot_path = f"outputs/plots/{args.scenario}_{run_timestamp}_reachability.svg"
        plot_func(model, labels, output_path=reachability_plot_path)
        print(f"Reachability plot saved to {reachability_plot_path}")

    clusters = extract_clusters(model, list(fields_by_mac.keys()))

    mac_to_label = extract_mac_to_label(df)

    clusters_without_noise = clusters
    if -1 in clusters_without_noise.keys():
        clusters_without_noise.pop(np.int64(-1))  # remove noise (cluster_id = -1)


    print_v_measure_report_simple(clusters_without_noise, mac_to_label)

    display_report(clusters, mac_to_label)
    if args.reverted_table:
        display_reverted_table(clusters, mac_to_label)
    validation_python(clusters_without_noise, mac_to_label)


    # Save report to file
    print("\nSaving report to file...")

    os.makedirs("outputs/reports", exist_ok=True)

    report_file = f"outputs/reports/{args.scenario}_{run_timestamp}.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print_v_measure_report_simple(clusters_without_noise, mac_to_label)
            display_report(clusters, mac_to_label)
            if args.reverted_table:
                display_reverted_table(clusters, mac_to_label)

    print(f"\nReport saved to {report_file}")

    csv_params = {
        "timestamp": run_timestamp,
    }

    if args.clustering_method == "optics":
        csv_params.update({
            "min_samples": args.min_samples,
            "xi": args.xi,
            "min_cluster_size": args.min_cluster_size,
            "max_eps": args.max_eps,
        })

    elif args.clustering_method == "dbscan":
        csv_params.update({
            "min_samples": args.min_samples,
            "eps": args.eps,
        })

    elif args.clustering_method == "hdbscan":
        csv_params.update({
            "min_samples": args.min_samples,
            "min_cluster_size": args.min_cluster_size,
            "cluster_selection_epsilon": args.cluster_selection_epsilon,
        })

    append_report_to_csv(
        report_file=report_file,
        csv_file=args.experiments_csv,
        scenario=args.scenario,
        clustering_method=args.clustering_method,
        params=csv_params,
    )

    print(f"Experiment row appended to {args.experiments_csv}")

    
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
