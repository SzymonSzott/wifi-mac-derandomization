import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


CSV_PATH = "outputs/experiments/5_hdbscan_italian_without_xiaomi.csv"
OUT_DIR = "outputs/experiments/analysis_hdbscan"

PLOT_EXT = "svg"

# Subset used only for thesis figures.
# Full analysis is still performed on all values.
PRESENTATION_CLUSTER_SELECTION_EPSILON_VALUES = [
    0.0,
    0.05,
    0.20,
    1.00,
]


AXIS_LABELS = {
    "min_cluster_size": "Min. cluster size",
    "min_samples": "Min. samples",
    "cluster_selection_epsilon": "Cluster selection epsilon",
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
    "incorrect_macs": "Incorrect MAC addresses",
    "score": "Combined score",
    "score_delta": "Combined score delta",
}

INTEGER_AXES = {
    "min_cluster_size",
    "min_samples",
    "num_clusters",
    "incorrect_macs",
}


def axis_label(name):
    return AXIS_LABELS.get(name, str(name).replace("_", " "))


def format_axis_value(value, axis_name):
    if axis_name in INTEGER_AXES and pd.notna(value):
        return str(int(round(float(value))))
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


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
            if len(row) == len(header) + 1 and "" in row:
                empty_idx = row.index("")
                row = row[:empty_idx] + row[empty_idx + 1:]

            if len(row) > len(header):
                row = row[:len(header)]

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
    890   -> 0.890
    933   -> 0.933
    0.890 -> 0.890
    """
    if pd.isna(value):
        return value

    if value > 1.0 and value <= 1000.0:
        return value / 1000.0

    return value


def save_lineplot(df, x, y, filename, title=None, y_min=None, y_max=None):
    if df.empty:
        print(f"SKIP empty lineplot: {filename}")
        return

    plot_df = df.dropna(subset=[x, y]).copy()

    if plot_df.empty:
        print(f"SKIP no valid data for lineplot: {filename}")
        return

    grouped = plot_df.groupby(x, as_index=False)[y].mean()
    grouped = grouped.sort_values(x)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(grouped[x], grouped[y], marker="o")
    ax.set_xlabel(axis_label(x))
    ax.set_ylabel(axis_label(y))

    if x in INTEGER_AXES:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if y in INTEGER_AXES:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)

    # No plot title: figures are titled in LaTeX captions.
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename))
    plt.close(fig)

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

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(pivot, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [format_axis_value(x, columns) for x in pivot.columns],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([format_axis_value(x, index) for x in pivot.index])

    ax.set_xlabel(axis_label(columns))
    ax.set_ylabel(axis_label(index))

    # No plot title: figures are titled in LaTeX captions.
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, format(value, fmt), ha="center", va="center")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(axis_label(values))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename))
    plt.close(fig)

    print(f"Saved: {filename}")


def save_heatmaps_for_cse(df, cse_value):
    subset = df[
        df["cluster_selection_epsilon"].round(8) == round(cse_value, 8)
    ].copy()

    suffix = str(cse_value).replace(".", "p")

    if subset.empty:
        print(f"SKIP heatmaps for cse={cse_value}: no rows")
        return

    save_heatmap(
        subset,
        index="min_samples",
        columns="min_cluster_size",
        values="score",
        filename=f"heatmap_cse{suffix}_ms_mcs_score.{PLOT_EXT}",
        title=f"HDBSCAN: combined score, cluster_selection_epsilon={cse_value}",
    )

    save_heatmap(
        subset,
        index="min_samples",
        columns="min_cluster_size",
        values="v_measure",
        filename=f"heatmap_cse{suffix}_ms_mcs_v_measure.{PLOT_EXT}",
        title=f"HDBSCAN: V-measure, cluster_selection_epsilon={cse_value}",
    )

    save_heatmap(
        subset,
        index="min_samples",
        columns="min_cluster_size",
        values="v_measure_delta",
        filename=f"heatmap_cse{suffix}_ms_mcs_v_measure_delta.{PLOT_EXT}",
        title=f"HDBSCAN: Δ V-measure from best, cluster_selection_epsilon={cse_value}",
    )

    save_heatmap(
        subset,
        index="min_samples",
        columns="min_cluster_size",
        values="noise_pct",
        filename=f"heatmap_cse{suffix}_ms_mcs_noise_pct.{PLOT_EXT}",
        title=f"HDBSCAN: noise %, cluster_selection_epsilon={cse_value}",
        fmt=".1f",
    )

    save_heatmap(
        subset,
        index="min_samples",
        columns="min_cluster_size",
        values="devices_one_cluster_pct",
        filename=f"heatmap_cse{suffix}_ms_mcs_devices_one_cluster_pct.{PLOT_EXT}",
        title=f"HDBSCAN: DeviceOneCluster %, cluster_selection_epsilon={cse_value}",
        fmt=".1f",
    )

    save_heatmap(
        subset,
        index="min_samples",
        columns="min_cluster_size",
        values="num_clusters",
        filename=f"heatmap_cse{suffix}_ms_mcs_num_clusters.{PLOT_EXT}",
        title=f"HDBSCAN: number of clusters, cluster_selection_epsilon={cse_value}",
        fmt=".0f",
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = read_mixed_csv(CSV_PATH)

    print(f"Loaded rows before cleaning: {len(df)}")

    numeric_cols = [
        "min_cluster_size",
        "min_samples",
        "cluster_selection_epsilon",
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

    # Degenerate or failed configurations may have empty metrics.
    # For analysis, treat them as failed configurations.
    for col in ["homogeneity", "completeness", "v_measure", "dcms_macro", "dcms_weighted"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    for col in ["noise_pct", "pure_clusters_pct", "devices_one_cluster_pct"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    df = df.drop_duplicates(
        subset=["min_cluster_size", "min_samples", "cluster_selection_epsilon"],
        keep="last",
    ).copy()

    print(f"Rows after removing duplicates: {len(df)}")
    print(
        "Unique configurations:",
        df[["min_cluster_size", "min_samples", "cluster_selection_epsilon"]]
        .drop_duplicates()
        .shape[0],
    )

    print("\nValue sanity check:")
    print("v_measure min/max:", df["v_measure"].min(), df["v_measure"].max())
    print("dcms_weighted min/max:", df["dcms_weighted"].min(), df["dcms_weighted"].max())
    print("min_cluster_size min/max:", df["min_cluster_size"].min(), df["min_cluster_size"].max())
    print("min_samples min/max:", df["min_samples"].min(), df["min_samples"].max())
    print(
        "cluster_selection_epsilon values:",
        sorted(df["cluster_selection_epsilon"].dropna().unique()),
    )

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
        df["cluster_selection_epsilon"]
        .round(8)
        .isin([round(x, 8) for x in PRESENTATION_CLUSTER_SELECTION_EPSILON_VALUES])
    ].copy()

    print(f"\nRows used for presentation plots: {len(presentation_df)}")
    print(
        "Presentation cluster_selection_epsilon values:",
        sorted(presentation_df["cluster_selection_epsilon"].dropna().unique()),
    )

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
        "min_cluster_size",
        "min_samples",
        "cluster_selection_epsilon",
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

    print("\n=== TOP 20 HDBSCAN CONFIGURATIONS ===")
    print(best[columns_to_show].head(20).to_string(index=False))

    print("\n=== TOP 20 FILTERED HDBSCAN CONFIGURATIONS ===")
    print(filtered[columns_to_show].head(20).to_string(index=False))

    # =========================
    # FULL LINE PLOTS
    # =========================

    save_lineplot(
        df,
        x="min_cluster_size",
        y="v_measure",
        filename=f"min_cluster_size_vs_v_measure.{PLOT_EXT}",
        title="HDBSCAN: V-measure vs min_cluster_size",
    )

    save_lineplot(
        df,
        x="min_cluster_size",
        y="noise_pct",
        filename=f"min_cluster_size_vs_noise_pct.{PLOT_EXT}",
        title="HDBSCAN: noise % vs min_cluster_size",
    )

    save_lineplot(
        df,
        x="min_cluster_size",
        y="devices_one_cluster_pct",
        filename=f"min_cluster_size_vs_devices_one_cluster_pct.{PLOT_EXT}",
        title="HDBSCAN: DeviceOneCluster % vs min_cluster_size",
    )

    save_lineplot(
        df,
        x="min_samples",
        y="v_measure",
        filename=f"min_samples_vs_v_measure.{PLOT_EXT}",
        title="HDBSCAN: V-measure vs min_samples",
    )

    save_lineplot(
        df,
        x="min_samples",
        y="noise_pct",
        filename=f"min_samples_vs_noise_pct.{PLOT_EXT}",
        title="HDBSCAN: noise % vs min_samples",
    )

    save_lineplot(
        df,
        x="cluster_selection_epsilon",
        y="v_measure",
        filename=f"cluster_selection_epsilon_vs_v_measure.{PLOT_EXT}",
        title="HDBSCAN: V-measure vs cluster_selection_epsilon",
    )

    save_lineplot(
        df,
        x="cluster_selection_epsilon",
        y="noise_pct",
        filename=f"cluster_selection_epsilon_vs_noise_pct.{PLOT_EXT}",
        title="HDBSCAN: noise % vs cluster_selection_epsilon",
    )

    save_lineplot(
        df,
        x="cluster_selection_epsilon",
        y="num_clusters",
        filename=f"cluster_selection_epsilon_vs_num_clusters.{PLOT_EXT}",
        title="HDBSCAN: number of clusters vs cluster_selection_epsilon",
    )

    # =========================
    # FULL GENERAL HEATMAPS
    # averaged over cluster_selection_epsilon
    # =========================

    save_heatmap(
        df,
        index="min_samples",
        columns="min_cluster_size",
        values="score",
        filename=f"heatmap_ms_mcs_score_mean_cse.{PLOT_EXT}",
        title="HDBSCAN: combined score averaged over cluster_selection_epsilon",
    )

    save_heatmap(
        df,
        index="min_samples",
        columns="min_cluster_size",
        values="v_measure",
        filename=f"heatmap_ms_mcs_v_measure_mean_cse.{PLOT_EXT}",
        title="HDBSCAN: V-measure averaged over cluster_selection_epsilon",
    )

    save_heatmap(
        df,
        index="min_samples",
        columns="min_cluster_size",
        values="noise_pct",
        filename=f"heatmap_ms_mcs_noise_pct_mean_cse.{PLOT_EXT}",
        title="HDBSCAN: noise % averaged over cluster_selection_epsilon",
        fmt=".1f",
    )

    save_heatmap(
        df,
        index="min_samples",
        columns="min_cluster_size",
        values="devices_one_cluster_pct",
        filename=f"heatmap_ms_mcs_devices_one_cluster_pct_mean_cse.{PLOT_EXT}",
        title="HDBSCAN: DeviceOneCluster % averaged over cluster_selection_epsilon",
        fmt=".1f",
    )

    # =========================
    # PRESENTATION PLOTS
    # These are intended for the thesis text.
    # =========================

    save_heatmap(
        presentation_df,
        index="min_samples",
        columns="min_cluster_size",
        values="score",
        filename=f"presentation_heatmap_ms_mcs_score_mean_cse.{PLOT_EXT}",
        title="HDBSCAN: combined score",
    )

    save_heatmap(
        presentation_df,
        index="min_samples",
        columns="min_cluster_size",
        values="v_measure",
        filename=f"presentation_heatmap_ms_mcs_v_measure_mean_cse.{PLOT_EXT}",
        title="HDBSCAN: V-measure",
    )

    save_heatmap(
        presentation_df,
        index="min_samples",
        columns="min_cluster_size",
        values="noise_pct",
        filename=f"presentation_heatmap_ms_mcs_noise_pct_mean_cse.{PLOT_EXT}",
        title="HDBSCAN: noise %",
        fmt=".1f",
    )

    save_heatmap(
        presentation_df,
        index="min_samples",
        columns="min_cluster_size",
        values="devices_one_cluster_pct",
        filename=f"presentation_heatmap_ms_mcs_devices_one_cluster_pct_mean_cse.{PLOT_EXT}",
        title="HDBSCAN: DeviceOneCluster %",
        fmt=".1f",
    )

    save_lineplot(
        presentation_df,
        x="min_cluster_size",
        y="v_measure",
        filename=f"presentation_min_cluster_size_vs_v_measure.{PLOT_EXT}",
        title="HDBSCAN: V-measure vs min_cluster_size",
    )

    save_lineplot(
        presentation_df,
        x="min_cluster_size",
        y="num_clusters",
        filename=f"presentation_min_cluster_size_vs_num_clusters.{PLOT_EXT}",
        title="HDBSCAN: number of clusters vs min_cluster_size",
    )

    save_lineplot(
        presentation_df,
        x="cluster_selection_epsilon",
        y="v_measure",
        filename=f"presentation_cluster_selection_epsilon_vs_v_measure.{PLOT_EXT}",
        title="HDBSCAN: V-measure vs cluster_selection_epsilon",
        y_min=0.0,
        y_max=1.0,
    )

    save_lineplot(
        presentation_df,
        x="cluster_selection_epsilon",
        y="num_clusters",
        filename=f"presentation_cluster_selection_epsilon_vs_num_clusters.{PLOT_EXT}",
        title="HDBSCAN: number of clusters vs cluster_selection_epsilon",
    )

    # =========================
    # HEATMAPS FOR EACH cluster_selection_epsilon
    # =========================

    cse_values = sorted(df["cluster_selection_epsilon"].dropna().unique())

    for cse_value in cse_values:
        save_heatmaps_for_cse(df, cse_value)

    print(f"\nSaved analysis to: {OUT_DIR}")
    print(f"Plot format: .{PLOT_EXT}")
    print(f"Best configurations: {best_path}")
    print(f"Best filtered configurations: {filtered_path}")


if __name__ == "__main__":
    main()