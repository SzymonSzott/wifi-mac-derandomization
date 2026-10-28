import os
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "outputs/experiments/3_optics_xi_wide.csv"
OUT_DIR = "outputs/experiments/analysis_optics_xi_wide"
PLOT_EXT = "svg"


AXIS_LABELS = {
    "min_samples": "min samples",
    "xi": "xi",
    "min_cluster_size": "min cluster size",
    "num_clusters": "No. of clusters",
    "v_measure": "V-measure",
    "v_measure_delta": "Δ V-measure from best",
    "noise_pct": "Noise [%]",
    "pure_clusters_pct": "Pure clusters [%]",
    "devices_one_cluster_pct": "DeviceOneCluster [%]",
    "dcms_weighted": "DCMS weighted",
    "score": "Combined score",
}


INTEGER_AXES = {
    "min_samples",
    "min_cluster_size",
    "num_clusters",
    "incorrect_macs",
}


def label_for(name):
    return AXIS_LABELS.get(name, name)


def format_tick(value, field_name):
    if field_name in INTEGER_AXES:
        try:
            return str(int(round(float(value))))
        except (TypeError, ValueError):
            return str(value)

    return str(value)


def save_heatmap(df, index, columns, values, filename, title=None, fmt=".3f"):
    plot_df = df.dropna(subset=[index, columns, values]).copy()

    if plot_df.empty:
        print(f"SKIP empty heatmap: {filename}")
        return

    pivot = plot_df.pivot_table(
        index=index,
        columns=columns,
        values=values,
        aggfunc="mean",
    )

    plt.figure(figsize=(13, 7))
    plt.imshow(pivot, aspect="auto")

    plt.xticks(
        range(len(pivot.columns)),
        [format_tick(x, columns) for x in pivot.columns],
        rotation=45,
    )
    plt.yticks(
        range(len(pivot.index)),
        [format_tick(x, index) for x in pivot.index],
    )

    plt.xlabel(label_for(columns))
    plt.ylabel(label_for(index))

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                plt.text(j, i, format(value, fmt), ha="center", va="center")

    plt.colorbar(label=label_for(values))
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename))
    plt.close()

    print(f"Saved: {filename}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    numeric_cols = [
        "min_samples",
        "xi",
        "min_cluster_size",
        "num_clusters",
        "v_measure",
        "noise_pct",
        "pure_clusters_pct",
        "devices_one_cluster_pct",
        "dcms_weighted",
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

    df["v_measure_delta"] = df["v_measure"].max() - df["v_measure"]
    df["score_delta"] = df["score"].max() - df["score"]

    best = df.sort_values(
        by=["score", "v_measure", "dcms_weighted"],
        ascending=False,
    )

    best_path = os.path.join(OUT_DIR, "best_configurations.csv")
    best.to_csv(best_path, index=False)

    print("\n=== TOP 20 XI-WIDE CONFIGURATIONS ===")
    print(
        best[
            [
                "scenario",
                "min_samples",
                "xi",
                "min_cluster_size",
                "num_clusters",
                "v_measure",
                "noise_pct",
                "devices_one_cluster_pct",
                "pure_clusters_pct",
                "dcms_weighted",
                "score",
                "report_file",
            ]
        ].head(20).to_string(index=False)
    )

    save_heatmap(
        df,
        index="min_samples",
        columns="xi",
        values="v_measure_delta",
        filename=f"heatmap_xiwide_ms_xi_v_measure_delta.{PLOT_EXT}",
        title="Δ V-measure from best: min_samples vs xi",
    )

    save_heatmap(
        df,
        index="min_samples",
        columns="xi",
        values="noise_pct",
        filename=f"heatmap_xiwide_ms_xi_noise_pct.{PLOT_EXT}",
        title="Noise %: min_samples vs xi",
        fmt=".1f",
    )

    save_heatmap(
        df,
        index="min_samples",
        columns="xi",
        values="score",
        filename=f"heatmap_xiwide_ms_xi_score.{PLOT_EXT}",
        title="Combined score: min_samples vs xi",
    )

    save_heatmap(
        df,
        index="min_samples",
        columns="xi",
        values="devices_one_cluster_pct",
        filename=f"heatmap_xiwide_ms_xi_devices_one_cluster_pct.{PLOT_EXT}",
        title="DeviceOneCluster %: min_samples vs xi",
        fmt=".1f",
    )

    print(f"\nSaved analysis to: {OUT_DIR}")
    print(f"Best configurations: {best_path}")


if __name__ == "__main__":
    main()
