import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


CSV_PATH = "outputs/experiments/4_dbscan_italian_without_xiaomi.csv"
OUT_DIR = "outputs/experiments/analysis_dbscan"

PLOT_EXT = "svg"

AXIS_LABELS = {
    "eps": "eps",
    "min_samples": "min samples",
    "num_clusters": "No. of clusters",
    "homogeneity": "Homogeneity",
    "completeness": "Completeness",
    "v_measure": "V-measure",
    "v_measure_delta": "V-measure delta",
    "noise_pct": "Noise [%]",
    "pure_clusters_pct": "Pure clusters [%]",
    "devices_one_cluster_pct": "DeviceOneCluster [%]",
    "dcms_macro": "DCMS macro",
    "dcms_weighted": "DCMS weighted",
    "incorrect_macs": "No. of incorrect MAC addresses",
    "score": "Combined score",
}

INTEGER_AXES = {
    "min_samples",
    "num_clusters",
    "incorrect_macs",
}


def axis_label(name):
    return AXIS_LABELS.get(name, name.replace("_", " "))


def format_tick(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if value.is_integer():
        return str(int(value))

    return f"{value:g}"


# Subset used only for figures included in the thesis.
# The full analysis is still performed on all eps values.
PRESENTATION_EPS_VALUES = [
    0.01,
    0.10,
    0.50,
    1.50,
    3.00,
    5.00,
    10.00,
    20.00,
    30.00,
    40.00,
    50.00,
]


def read_mixed_csv(path):
    """
    Reads CSV robustly even if some rows have one extra empty column
    caused by comma decimal separators.
    """
    rows = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            # Fix rows that accidentally contain one extra empty column.
            if len(row) == len(header) + 1 and "" in row:
                empty_idx = row.index("")
                row = row[:empty_idx] + row[empty_idx + 1:]

            # Trim too-long rows.
            if len(row) > len(header):
                row = row[:len(header)]

            # Pad too-short rows.
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))

            rows.append(row)

    return pd.DataFrame(rows, columns=header)


def parse_number(value):
    if pd.isna(value):
        return float("nan")

    text = str(value).strip()

    if text == "":
        return float("nan")

    text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return float("nan")


def fix_unit_metric(value):
    """
    Fix metrics that should be in range 0-1.

    Examples:
    872   -> 0.872
    919   -> 0.919
    0.872 -> 0.872
    """
    if pd.isna(value):
        return value

    if value > 1.0 and value <= 1000.0:
        return value / 1000.0

    return value


def save_lineplot(df, x, y, filename, title=None):
    if df.empty:
        print(f"SKIP empty lineplot: {filename}")
        return

    plot_df = df.dropna(subset=[x, y]).copy()

    if plot_df.empty:
        print(f"SKIP no valid data for lineplot: {filename}")
        return

    plt.figure(figsize=(9, 5))
    ax = plt.gca()

    grouped = plot_df.groupby(x, as_index=False)[y].mean()
    grouped = grouped.sort_values(x)

    ax.plot(grouped[x], grouped[y], marker="o")
    ax.set_xlabel(axis_label(x))
    ax.set_ylabel(axis_label(y))

    # No plot title is added, because figures are described in LaTeX captions.
    if x in INTEGER_AXES:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if y in INTEGER_AXES:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename))
    plt.close()

    print(f"Saved: {filename}")

def save_heatmap(df, index, columns, values, filename, title=None, fmt=".3f"):
    if df.empty:
        print(f"SKIP empty heatmap: {filename}")
        return

    plot_df = df.dropna(subset=[index, columns, values]).copy()

    if plot_df.empty:
        print(f"SKIP no valid data for heatmap: {filename}")
        return

    pivot = plot_df.pivot_table(
        index=index,
        columns=columns,
        values=values,
        aggfunc="mean",
    )

    if pivot.empty:
        print(f"SKIP empty pivot: {filename}")
        return

    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    image = ax.imshow(pivot, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([format_tick(x) for x in pivot.columns], rotation=45)

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([format_tick(x) for x in pivot.index])

    ax.set_xlabel(axis_label(columns))
    ax.set_ylabel(axis_label(index))

    # No plot title is added, because figures are described in LaTeX captions.
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, format(value, fmt), ha="center", va="center")

    cbar = plt.colorbar(image, label=axis_label(values))
    if values in INTEGER_AXES:
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename))
    plt.close()

    print(f"Saved: {filename}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = read_mixed_csv(CSV_PATH)

    print(f"Loaded rows before cleaning: {len(df)}")

    numeric_cols = [
        "eps",
        "min_samples",
        "num_clusters",
        "homogeneity",
        "completeness",
        "v_measure",
        "noise_pct",
        "pure_clusters_pct",
        "devices_one_cluster_pct",
        "dcms_macro",
        "dcms_weighted",
        "incorrect_macs",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_number)

    unit_metric_cols = [
        "homogeneity",
        "completeness",
        "v_measure",
        "dcms_macro",
        "dcms_weighted",
    ]

    for col in unit_metric_cols:
        if col in df.columns:
            df[col] = df[col].apply(fix_unit_metric)

    # Degenerate cases, e.g. one cluster, may have empty V-measure fields.
    # For analysis we treat them as failed configurations.
    for col in ["homogeneity", "completeness", "v_measure", "dcms_macro", "dcms_weighted"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    for col in ["noise_pct", "pure_clusters_pct", "devices_one_cluster_pct"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    df = df.drop_duplicates(
        subset=["eps", "min_samples"],
        keep="last",
    ).copy()

    print(f"Rows after removing duplicates: {len(df)}")
    print(f"Unique configurations: {df[['eps', 'min_samples']].drop_duplicates().shape[0]}")

    print("\nValue sanity check:")
    print("v_measure min/max:", df["v_measure"].min(), df["v_measure"].max())
    print("dcms_weighted min/max:", df["dcms_weighted"].min(), df["dcms_weighted"].max())
    print("eps min/max:", df["eps"].min(), df["eps"].max())

    df["score"] = (
        0.40 * df["v_measure"] +
        0.25 * df["dcms_weighted"] +
        0.20 * (df["devices_one_cluster_pct"] / 100.0) +
        0.15 * (df["pure_clusters_pct"] / 100.0) -
        0.10 * (df["noise_pct"] / 100.0)
    )

    df["v_measure_delta"] = df["v_measure"].max() - df["v_measure"]
    df["score_delta"] = df["score"].max() - df["score"]

    presentation_df = df[
        df["eps"].round(8).isin([round(x, 8) for x in PRESENTATION_EPS_VALUES])
    ].copy()

    print(f"\nRows used for presentation plots: {len(presentation_df)}")
    print("Presentation eps values:", sorted(presentation_df["eps"].dropna().unique()))

    best = df.sort_values(
        by=["score", "v_measure", "dcms_weighted"],
        ascending=False,
    )

    best_path = os.path.join(OUT_DIR, "best_configurations.csv")
    best.to_csv(best_path, index=False)

    filtered = df[
        (df["noise_pct"] <= 30.0) &
        (df["num_clusters"] >= 2) &
        (df["v_measure"] >= 0.50) &
        (df["dcms_weighted"] >= 0.60)
    ].copy()

    filtered = filtered.sort_values(
        by=["score", "v_measure", "dcms_weighted"],
        ascending=False,
    )

    filtered_path = os.path.join(OUT_DIR, "best_filtered_configurations.csv")
    filtered.to_csv(filtered_path, index=False)

    columns_to_show = [
        "scenario",
        "eps",
        "min_samples",
        "num_clusters",
        "v_measure",
        "noise_pct",
        "devices_one_cluster_pct",
        "pure_clusters_pct",
        "dcms_macro",
        "dcms_weighted",
        "score",
        "report_file",
    ]

    columns_to_show = [col for col in columns_to_show if col in df.columns]

    print("\n=== TOP 20 CONFIGURATIONS ===")
    print(best[columns_to_show].head(20).to_string(index=False))

    print("\n=== TOP 20 FILTERED CONFIGURATIONS ===")
    print(filtered[columns_to_show].head(20).to_string(index=False))

    # =========================
    # FULL LINE PLOTS
    # =========================

    save_lineplot(
        df,
        x="eps",
        y="v_measure",
        filename=f"eps_vs_v_measure.{PLOT_EXT}",
        title="DBSCAN: V-measure vs eps",
    )

    save_lineplot(
        df,
        x="eps",
        y="noise_pct",
        filename=f"eps_vs_noise_pct.{PLOT_EXT}",
        title="DBSCAN: noise % vs eps",
    )

    save_lineplot(
        df,
        x="eps",
        y="num_clusters",
        filename=f"eps_vs_num_clusters.{PLOT_EXT}",
        title="DBSCAN: number of clusters vs eps",
    )

    save_lineplot(
        df,
        x="eps",
        y="dcms_weighted",
        filename=f"eps_vs_dcms_weighted.{PLOT_EXT}",
        title="DBSCAN: DCMS weighted vs eps",
    )

    save_lineplot(
        df,
        x="eps",
        y="devices_one_cluster_pct",
        filename=f"eps_vs_devices_one_cluster_pct.{PLOT_EXT}",
        title="DBSCAN: DeviceOneCluster % vs eps",
    )

    save_lineplot(
        df,
        x="min_samples",
        y="v_measure",
        filename=f"min_samples_vs_v_measure.{PLOT_EXT}",
        title="DBSCAN: V-measure vs min_samples",
    )

    save_lineplot(
        df,
        x="min_samples",
        y="noise_pct",
        filename=f"min_samples_vs_noise_pct.{PLOT_EXT}",
        title="DBSCAN: noise % vs min_samples",
    )

    save_lineplot(
        df,
        x="min_samples",
        y="num_clusters",
        filename=f"min_samples_vs_num_clusters.{PLOT_EXT}",
        title="DBSCAN: number of clusters vs min_samples",
    )

    # =========================
    # FULL HEATMAPS
    # =========================

    save_heatmap(
        df,
        index="min_samples",
        columns="eps",
        values="score",
        filename=f"heatmap_eps_ms_score.{PLOT_EXT}",
        title="DBSCAN: combined score",
    )

    save_heatmap(
        df,
        index="min_samples",
        columns="eps",
        values="v_measure",
        filename=f"heatmap_eps_ms_v_measure.{PLOT_EXT}",
        title="DBSCAN: V-measure",
    )

    save_heatmap(
        df,
        index="min_samples",
        columns="eps",
        values="v_measure_delta",
        filename=f"heatmap_eps_ms_v_measure_delta.{PLOT_EXT}",
        title="DBSCAN: Δ V-measure from best",
    )

    save_heatmap(
        df,
        index="min_samples",
        columns="eps",
        values="noise_pct",
        filename=f"heatmap_eps_ms_noise_pct.{PLOT_EXT}",
        title="DBSCAN: noise %",
        fmt=".1f",
    )

    save_heatmap(
        df,
        index="min_samples",
        columns="eps",
        values="dcms_weighted",
        filename=f"heatmap_eps_ms_dcms_weighted.{PLOT_EXT}",
        title="DBSCAN: DCMS weighted",
    )

    save_heatmap(
        df,
        index="min_samples",
        columns="eps",
        values="devices_one_cluster_pct",
        filename=f"heatmap_eps_ms_devices_one_cluster_pct.{PLOT_EXT}",
        title="DBSCAN: DeviceOneCluster %",
        fmt=".1f",
    )

    save_heatmap(
        df,
        index="min_samples",
        columns="eps",
        values="num_clusters",
        filename=f"heatmap_eps_ms_num_clusters.{PLOT_EXT}",
        title="DBSCAN: number of clusters",
        fmt=".0f",
    )

    # =========================
    # PRESENTATION PLOTS
    # These use a reduced eps subset for thesis readability.
    # =========================

    save_heatmap(
        presentation_df,
        index="min_samples",
        columns="eps",
        values="score",
        filename=f"presentation_heatmap_eps_ms_score.{PLOT_EXT}",
        title="DBSCAN: combined score",
    )

    save_heatmap(
        presentation_df,
        index="min_samples",
        columns="eps",
        values="num_clusters",
        filename=f"presentation_heatmap_eps_ms_num_clusters.{PLOT_EXT}",
        title="DBSCAN: number of clusters",
        fmt=".0f",
    )

    save_heatmap(
        presentation_df,
        index="min_samples",
        columns="eps",
        values="noise_pct",
        filename=f"presentation_heatmap_eps_ms_noise_pct.{PLOT_EXT}",
        title="DBSCAN: noise %",
        fmt=".1f",
    )

    save_lineplot(
        presentation_df,
        x="eps",
        y="num_clusters",
        filename=f"presentation_eps_vs_num_clusters.{PLOT_EXT}",
        title="DBSCAN: number of clusters vs eps",
    )

    save_lineplot(
        presentation_df,
        x="eps",
        y="v_measure",
        filename=f"presentation_eps_vs_v_measure.{PLOT_EXT}",
        title="DBSCAN: V-measure vs eps",
    )

    save_lineplot(
        presentation_df,
        x="eps",
        y="noise_pct",
        filename=f"presentation_eps_vs_noise_pct.{PLOT_EXT}",
        title="DBSCAN: noise % vs eps",
    )

    print(f"\nSaved analysis to: {OUT_DIR}")
    print(f"Plot format: .{PLOT_EXT}")
    print(f"Best configurations: {best_path}")
    print(f"Best filtered configurations: {filtered_path}")


if __name__ == "__main__":
    main()