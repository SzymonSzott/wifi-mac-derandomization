import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


CSV_PATH = "outputs/experiments/3_optics_italian_without_xiaomi.csv"
OUT_DIR = "outputs/experiments/analysis"

PLOT_EXT = "svg"


AXIS_LABELS = {
    "min_samples": "Min. samples",
    "xi": "xi",
    "min_cluster_size": "Min. cluster size",
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
    "score_delta": "Combined score delta",
}

INTEGER_AXES = {
    "min_samples",
    "min_cluster_size",
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


def save_lineplot(df, x, y, filename, title=None):
    if df.empty:
        print(f"SKIP empty lineplot: {filename}")
        return

    if x not in df.columns or y not in df.columns:
        print(f"SKIP missing columns for lineplot: {filename}")
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

    # No plot title is added, because figures are described in LaTeX captions.
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename))
    plt.close(fig)

    print(f"Saved: {filename}")


def save_heatmap(
    df,
    index,
    columns,
    values,
    filename,
    title=None,
    aggfunc="mean",
    fmt=".3f",
):
    if df.empty:
        print(f"SKIP empty heatmap: {filename}")
        return

    required = {index, columns, values}
    if not required.issubset(df.columns):
        print(f"SKIP missing columns for heatmap: {filename}")
        return

    plot_df = df.dropna(subset=[index, columns, values]).copy()

    if plot_df.empty:
        print(f"SKIP no valid data for heatmap: {filename}")
        return

    pivot = plot_df.pivot_table(
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
    )

    if pivot.empty:
        print(f"SKIP empty pivot for heatmap: {filename}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [format_axis_value(x, columns) for x in pivot.columns],
        rotation=45,
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(
        [format_axis_value(x, index) for x in pivot.index],
    )

    ax.set_xlabel(axis_label(columns))
    ax.set_ylabel(axis_label(index))

    # No plot title is added, because figures are described in LaTeX captions.
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, format(value, fmt), ha="center", va="center")

    fig.colorbar(im, ax=ax, label=axis_label(values))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename))
    plt.close(fig)

    print(f"Saved: {filename}")


def save_best_tables(df):
    best = df.sort_values(
        by=["score", "v_measure", "dcms_weighted"],
        ascending=False,
    )

    best_path = os.path.join(OUT_DIR, "best_configurations.csv")
    best.to_csv(best_path, index=False)

    filtered = df[
        (df["noise_pct"] <= 25.0) &
        (df["num_clusters"] >= 2) &
        (df["v_measure"] >= 0.70) &
        (df["dcms_weighted"] >= 0.80)
    ].copy()

    filtered = filtered.sort_values(
        by=["score", "v_measure", "dcms_weighted"],
        ascending=False,
    )

    filtered_path = os.path.join(OUT_DIR, "best_filtered_configurations.csv")
    filtered.to_csv(filtered_path, index=False)

    columns_to_show = [
        "scenario",
        "min_samples",
        "xi",
        "min_cluster_size",
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

    return best_path, filtered_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    numeric_cols = [
        "min_samples",
        "xi",
        "min_cluster_size",
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
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["v_measure"]).copy()

    df["score"] = (
        0.40 * df["v_measure"] +
        0.25 * df["dcms_weighted"] +
        0.20 * (df["devices_one_cluster_pct"] / 100.0) +
        0.15 * (df["pure_clusters_pct"] / 100.0) -
        0.10 * (df["noise_pct"] / 100.0)
    )

    best_path, filtered_path = save_best_tables(df)

    # =========================
    # SPLIT INTO EXPERIMENT TYPES
    # =========================

    scenario = df["scenario"].astype(str)

    onefactor_min_samples_df = df[
        scenario.str.startswith("onefactor_min_samples")
    ].copy()

    onefactor_xi_df = df[
        scenario.str.startswith("onefactor_xi")
    ].copy()

    onefactor_mcs_df = df[
        scenario.str.startswith("onefactor_mcs")
    ].copy()

    grid_df = df[
        scenario.str.startswith("grid_")
    ].copy()

    if grid_df.empty:
        print("WARNING: grid_df is empty. Heatmaps will use all data as fallback.")
        grid_df = df.copy()

    grid_df["v_measure_delta"] = grid_df["v_measure"].max() - grid_df["v_measure"]
    grid_df["score_delta"] = grid_df["score"].max() - grid_df["score"]

    grid_df_mcs15 = grid_df[
        grid_df["min_cluster_size"] == 15
    ].copy()

    if grid_df_mcs15.empty:
        print("WARNING: grid_df_mcs15 is empty. Fixed-mcs heatmaps will be skipped.")

    # =========================
    # 1D PLOTS — ONE FACTOR
    # =========================

    save_lineplot(
        onefactor_min_samples_df,
        x="min_samples",
        y="v_measure",
        filename=f"onefactor_min_samples_vs_v_measure.{PLOT_EXT}",
        title="One-factor: V-measure vs min_samples",
    )

    save_lineplot(
        onefactor_min_samples_df,
        x="min_samples",
        y="noise_pct",
        filename=f"onefactor_min_samples_vs_noise_pct.{PLOT_EXT}",
        title="One-factor: noise % vs min_samples",
    )

    save_lineplot(
        onefactor_min_samples_df,
        x="min_samples",
        y="num_clusters",
        filename=f"onefactor_min_samples_vs_num_clusters.{PLOT_EXT}",
        title="One-factor: number of clusters vs min_samples",
    )

    save_lineplot(
        onefactor_min_samples_df,
        x="min_samples",
        y="devices_one_cluster_pct",
        filename=f"onefactor_min_samples_vs_devices_one_cluster_pct.{PLOT_EXT}",
        title="One-factor: DeviceOneCluster % vs min_samples",
    )

    save_lineplot(
        onefactor_min_samples_df,
        x="min_samples",
        y="dcms_weighted",
        filename=f"onefactor_min_samples_vs_dcms_weighted.{PLOT_EXT}",
        title="One-factor: DCMS weighted vs min_samples",
    )

    save_lineplot(
        onefactor_xi_df,
        x="xi",
        y="v_measure",
        filename=f"onefactor_xi_vs_v_measure.{PLOT_EXT}",
        title="One-factor: V-measure vs xi",
    )

    save_lineplot(
        onefactor_xi_df,
        x="xi",
        y="noise_pct",
        filename=f"onefactor_xi_vs_noise_pct.{PLOT_EXT}",
        title="One-factor: noise % vs xi",
    )

    save_lineplot(
        onefactor_xi_df,
        x="xi",
        y="num_clusters",
        filename=f"onefactor_xi_vs_num_clusters.{PLOT_EXT}",
        title="One-factor: number of clusters vs xi",
    )

    save_lineplot(
        onefactor_xi_df,
        x="xi",
        y="devices_one_cluster_pct",
        filename=f"onefactor_xi_vs_devices_one_cluster_pct.{PLOT_EXT}",
        title="One-factor: DeviceOneCluster % vs xi",
    )

    save_lineplot(
        onefactor_xi_df,
        x="xi",
        y="dcms_weighted",
        filename=f"onefactor_xi_vs_dcms_weighted.{PLOT_EXT}",
        title="One-factor: DCMS weighted vs xi",
    )

    save_lineplot(
        onefactor_mcs_df,
        x="min_cluster_size",
        y="v_measure",
        filename=f"onefactor_min_cluster_size_vs_v_measure.{PLOT_EXT}",
        title="One-factor: V-measure vs min_cluster_size",
    )

    save_lineplot(
        onefactor_mcs_df,
        x="min_cluster_size",
        y="noise_pct",
        filename=f"onefactor_min_cluster_size_vs_noise_pct.{PLOT_EXT}",
        title="One-factor: noise % vs min_cluster_size",
    )

    save_lineplot(
        onefactor_mcs_df,
        x="min_cluster_size",
        y="num_clusters",
        filename=f"onefactor_min_cluster_size_vs_num_clusters.{PLOT_EXT}",
        title="One-factor: number of clusters vs min_cluster_size",
    )

    save_lineplot(
        onefactor_mcs_df,
        x="min_cluster_size",
        y="devices_one_cluster_pct",
        filename=f"onefactor_min_cluster_size_vs_devices_one_cluster_pct.{PLOT_EXT}",
        title="One-factor: DeviceOneCluster % vs min_cluster_size",
    )

    save_lineplot(
        onefactor_mcs_df,
        x="min_cluster_size",
        y="dcms_weighted",
        filename=f"onefactor_min_cluster_size_vs_dcms_weighted.{PLOT_EXT}",
        title="One-factor: DCMS weighted vs min_cluster_size",
    )

    # =========================
    # 1D PLOTS — GRID MEAN
    # =========================
    # These plots show the mean over the remaining parameters in the grid.

    save_lineplot(
        grid_df,
        x="min_samples",
        y="v_measure",
        filename=f"grid_mean_min_samples_vs_v_measure.{PLOT_EXT}",
        title="Grid mean: V-measure vs min_samples",
    )

    save_lineplot(
        grid_df,
        x="xi",
        y="v_measure",
        filename=f"grid_mean_xi_vs_v_measure.{PLOT_EXT}",
        title="Grid mean: V-measure vs xi",
    )

    save_lineplot(
        grid_df,
        x="min_cluster_size",
        y="v_measure",
        filename=f"grid_mean_min_cluster_size_vs_v_measure.{PLOT_EXT}",
        title="Grid mean: V-measure vs min_cluster_size",
    )

    save_lineplot(
        grid_df,
        x="min_samples",
        y="noise_pct",
        filename=f"grid_mean_min_samples_vs_noise_pct.{PLOT_EXT}",
        title="Grid mean: noise % vs min_samples",
    )

    save_lineplot(
        grid_df,
        x="xi",
        y="noise_pct",
        filename=f"grid_mean_xi_vs_noise_pct.{PLOT_EXT}",
        title="Grid mean: noise % vs xi",
    )

    save_lineplot(
        grid_df,
        x="min_cluster_size",
        y="noise_pct",
        filename=f"grid_mean_min_cluster_size_vs_noise_pct.{PLOT_EXT}",
        title="Grid mean: noise % vs min_cluster_size",
    )

    save_lineplot(
        grid_df,
        x="min_samples",
        y="dcms_weighted",
        filename=f"grid_mean_min_samples_vs_dcms_weighted.{PLOT_EXT}",
        title="Grid mean: DCMS weighted vs min_samples",
    )

    save_lineplot(
        grid_df,
        x="xi",
        y="dcms_weighted",
        filename=f"grid_mean_xi_vs_dcms_weighted.{PLOT_EXT}",
        title="Grid mean: DCMS weighted vs xi",
    )

    save_lineplot(
        grid_df,
        x="min_cluster_size",
        y="dcms_weighted",
        filename=f"grid_mean_min_cluster_size_vs_dcms_weighted.{PLOT_EXT}",
        title="Grid mean: DCMS weighted vs min_cluster_size",
    )

    # =========================
    # HEATMAPS — GRID MEAN OVER min_cluster_size
    # =========================
    # Here each cell is the mean over different min_cluster_size values.

    save_heatmap(
        grid_df,
        index="min_samples",
        columns="xi",
        values="v_measure",
        filename=f"heatmap_grid_mean_mcs_min_samples_xi_v_measure.{PLOT_EXT}",
        title="Grid mean over min_cluster_size: V-measure",
    )

    save_heatmap(
        grid_df,
        index="min_samples",
        columns="xi",
        values="v_measure_delta",
        filename=f"heatmap_grid_mean_mcs_min_samples_xi_v_measure_delta.{PLOT_EXT}",
        title="Grid mean over min_cluster_size: Δ V-measure from best",
    )

    save_heatmap(
        grid_df,
        index="min_samples",
        columns="xi",
        values="noise_pct",
        filename=f"heatmap_grid_mean_mcs_min_samples_xi_noise_pct.{PLOT_EXT}",
        title="Grid mean over min_cluster_size: noise %",
    )

    save_heatmap(
        grid_df,
        index="min_samples",
        columns="xi",
        values="dcms_weighted",
        filename=f"heatmap_grid_mean_mcs_min_samples_xi_dcms_weighted.{PLOT_EXT}",
        title="Grid mean over min_cluster_size: DCMS weighted",
    )

    save_heatmap(
        grid_df,
        index="min_samples",
        columns="xi",
        values="devices_one_cluster_pct",
        filename=f"heatmap_grid_mean_mcs_min_samples_xi_devices_one_cluster_pct.{PLOT_EXT}",
        title="Grid mean over min_cluster_size: DeviceOneCluster %",
    )

    save_heatmap(
        grid_df,
        index="min_samples",
        columns="xi",
        values="pure_clusters_pct",
        filename=f"heatmap_grid_mean_mcs_min_samples_xi_pure_clusters_pct.{PLOT_EXT}",
        title="Grid mean over min_cluster_size: pure clusters %",
    )

    save_heatmap(
        grid_df,
        index="min_samples",
        columns="xi",
        values="score",
        filename=f"heatmap_grid_mean_mcs_min_samples_xi_score.{PLOT_EXT}",
        title="Grid mean over min_cluster_size: score",
    )

    save_heatmap(
        grid_df,
        index="min_samples",
        columns="xi",
        values="score_delta",
        filename=f"heatmap_grid_mean_mcs_min_samples_xi_score_delta.{PLOT_EXT}",
        title="Grid mean over min_cluster_size: Δ score from best",
    )

    # =========================
    # HEATMAPS — FIXED min_cluster_size = 15
    # =========================
    # This is the most important version for the thesis, because it matches the baseline.

    if not grid_df_mcs15.empty:
        grid_df_mcs15["v_measure_delta"] = (
            grid_df_mcs15["v_measure"].max() - grid_df_mcs15["v_measure"]
        )
        grid_df_mcs15["score_delta"] = (
            grid_df_mcs15["score"].max() - grid_df_mcs15["score"]
        )

        save_heatmap(
            grid_df_mcs15,
            index="min_samples",
            columns="xi",
            values="v_measure",
            filename=f"heatmap_grid_mcs15_min_samples_xi_v_measure.{PLOT_EXT}",
            title="min_cluster_size=15: V-measure",
        )

        save_heatmap(
            grid_df_mcs15,
            index="min_samples",
            columns="xi",
            values="v_measure_delta",
            filename=f"heatmap_grid_mcs15_min_samples_xi_v_measure_delta.{PLOT_EXT}",
            title="min_cluster_size=15: Δ V-measure from best",
        )

        save_heatmap(
            grid_df_mcs15,
            index="min_samples",
            columns="xi",
            values="noise_pct",
            filename=f"heatmap_grid_mcs15_min_samples_xi_noise_pct.{PLOT_EXT}",
            title="min_cluster_size=15: noise %",
        )

        save_heatmap(
            grid_df_mcs15,
            index="min_samples",
            columns="xi",
            values="dcms_weighted",
            filename=f"heatmap_grid_mcs15_min_samples_xi_dcms_weighted.{PLOT_EXT}",
            title="min_cluster_size=15: DCMS weighted",
        )

        save_heatmap(
            grid_df_mcs15,
            index="min_samples",
            columns="xi",
            values="devices_one_cluster_pct",
            filename=f"heatmap_grid_mcs15_min_samples_xi_devices_one_cluster_pct.{PLOT_EXT}",
            title="min_cluster_size=15: DeviceOneCluster %",
        )

        save_heatmap(
            grid_df_mcs15,
            index="min_samples",
            columns="xi",
            values="pure_clusters_pct",
            filename=f"heatmap_grid_mcs15_min_samples_xi_pure_clusters_pct.{PLOT_EXT}",
            title="min_cluster_size=15: pure clusters %",
        )

        save_heatmap(
            grid_df_mcs15,
            index="min_samples",
            columns="xi",
            values="score",
            filename=f"heatmap_grid_mcs15_min_samples_xi_score.{PLOT_EXT}",
            title="min_cluster_size=15: score",
        )

        save_heatmap(
            grid_df_mcs15,
            index="min_samples",
            columns="xi",
            values="score_delta",
            filename=f"heatmap_grid_mcs15_min_samples_xi_score_delta.{PLOT_EXT}",
            title="min_cluster_size=15: Δ score from best",
        )

    print(f"\nSaved analysis to: {OUT_DIR}")
    print(f"Plot format: .{PLOT_EXT}")
    print(f"Best configurations: {best_path}")
    print(f"Best filtered configurations: {filtered_path}")

    print("\nGenerated groups:")
    print(f"onefactor_min_samples rows: {len(onefactor_min_samples_df)}")
    print(f"onefactor_xi rows: {len(onefactor_xi_df)}")
    print(f"onefactor_mcs rows: {len(onefactor_mcs_df)}")
    print(f"grid rows: {len(grid_df)}")
    print(f"grid mcs=15 rows: {len(grid_df_mcs15)}")


if __name__ == "__main__":
    main()