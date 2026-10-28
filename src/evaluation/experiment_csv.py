# evaluation/experiment_csv.py

import csv
import os
import re
from datetime import datetime


CSV_FIELDNAMES = [
    "timestamp",
    "scenario",
    "clustering_method",

    "min_samples",
    "eps",
    "xi",
    "min_cluster_size",
    "max_eps",
    "cluster_selection_epsilon",

    "n",
    "num_classes",
    "num_clusters",
    "homogeneity",
    "completeness",
    "v_measure",

    "correct_macs",
    "incorrect_macs",
    "total_analyzed_macs",

    "noise_pct",
    "noise_count",
    "noise_total",

    "labeled_clusters",
    "pure_clusters",
    "pure_clusters_pct",

    "devices_one_cluster",
    "devices_total",
    "devices_one_cluster_pct",

    "dcms_labeled_clusters",
    "dcms_macro",
    "dcms_weighted",
    "ideal_matches",

    "report_file",
]


def _search(pattern, text, cast=str, group=1, default=None):
    match = re.search(pattern, text)
    if not match:
        return default
    try:
        return cast(match.group(group))
    except Exception:
        return default


def parse_report_text(text):
    """
    Parses an existing text report without changing the display_report() function.
    Returns a dict of metrics for a single CSV row.
    """

    row = {}

    # --- V-MEASURE ---
    m = re.search(
        r"N=(\d+),\s*#classes=(\d+),\s*#clusters=(\d+)",
        text
    )
    if m:
        row["n"] = int(m.group(1))
        row["num_classes"] = int(m.group(2))
        row["num_clusters"] = int(m.group(3))

    m = re.search(
        r"h=([0-9.]+)\s+c=([0-9.]+)\s+V-measure\s*\(β=[0-9.]+\)\s*=\s*([0-9.]+)",
        text
    )
    if m:
        row["homogeneity"] = float(m.group(1))
        row["completeness"] = float(m.group(2))
        row["v_measure"] = float(m.group(3))

    # --- CLUSTER VALIDATION ---
    row["correct_macs"] = _search(
        r"Correctly classified MACs:\s*(\d+)",
        text,
        int
    )

    row["incorrect_macs"] = _search(
        r"Incorrectly classified MACs:\s*(\d+)",
        text,
        int
    )

    row["total_analyzed_macs"] = _search(
        r"Total number of analyzed MACs:\s*(\d+)",
        text,
        int
    )

    # --- NOISE ---
    m = re.search(
        r"Percentage of MACs classified as noise:\s*([0-9.]+)%\s*\((\d+)\s*of\s*(\d+)\)",
        text
    )
    if m:
        row["noise_pct"] = float(m.group(1))
        row["noise_count"] = int(m.group(2))
        row["noise_total"] = int(m.group(3))

    # --- SUCCESS METRICS ---
    matches = re.findall(
        r"Clusters with labeled MACs:\s*(\d+)",
        text
    )
    if matches:
        row["labeled_clusters"] = int(matches[0])
        if len(matches) > 1:
            row["dcms_labeled_clusters"] = int(matches[1])

    m = re.search(
        r"Pure clusters \(single LABEL\):\s*(\d+)\s*\(([0-9.]+)%\)",
        text
    )
    if m:
        row["pure_clusters"] = int(m.group(1))
        row["pure_clusters_pct"] = float(m.group(2))

    m = re.search(
        r"Devices recognized as a single device:\s*(\d+)/(\d+)\s*\(([0-9.]+)%\)",
        text
    )
    if m:
        row["devices_one_cluster"] = int(m.group(1))
        row["devices_total"] = int(m.group(2))
        row["devices_one_cluster_pct"] = float(m.group(3))

    # --- DCMS ---
    row["dcms_macro"] = _search(
        r"Average DCMS \(macro\):\s*([0-9.]+)",
        text,
        float
    )

    row["dcms_weighted"] = _search(
        r"Average DCMS \(weighted by \|C\|\):\s*([0-9.]+)",
        text,
        float
    )

    row["ideal_matches"] = _search(
        r"Perfect matches \(DCMS=1 and C==D\):\s*(\d+)",
        text,
        int
    )

    return row


def append_report_to_csv(
    report_file,
    csv_file,
    scenario,
    clustering_method,
    params=None,
):
    """
    Reads the saved .txt report and appends one row to the CSV.
    Changes nothing in the existing reports.
    """

    params = params or {}

    with open(report_file, "r", encoding="utf-8") as f:
        text = f.read()

    row = parse_report_text(text)

    row.update({
        "timestamp": params.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S"),
        "scenario": scenario,
        "clustering_method": clustering_method,
        "report_file": report_file,

        "min_samples": params.get("min_samples"),
        "eps": params.get("eps"),
        "xi": params.get("xi"),
        "min_cluster_size": params.get("min_cluster_size"),
        "max_eps": params.get("max_eps"),
        "cluster_selection_epsilon": params.get("cluster_selection_epsilon"),
    })

    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    file_exists = os.path.exists(csv_file)

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)

        if not file_exists:
            writer.writeheader()

        writer.writerow({field: row.get(field) for field in CSV_FIELDNAMES})