#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency


DEFAULT_EXPERIMENT = "moco3"
DEFAULT_MODEL = "ViViT"
DEFAULT_EPOCH = 100
BASE_COARSE_OUTPUT_DIR = Path("./VideoContrastive/data/analysis/analysis_output_coarse")


FEATURE_SETS = {
    "v1": [
        "accident_place_coarse",
        "accident_place_feature_coarse",
        "accident_negligence_rateA",
        "accident_negligence_rateB",
        "vehicle_a_progress_info_coarse_v1",
        "vehicle_b_progress_info_coarse_v1",
    ],
    "v2": [
        "accident_place_coarse",
        "accident_place_feature_coarse",
        "accident_negligence_rateA",
        "accident_negligence_rateB",
        "vehicle_a_progress_info_coarse_v2",
        "vehicle_b_progress_info_coarse_v2",
    ],
}


def parse_args():
    p = argparse.ArgumentParser(description="First-order coarse analysis.")
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT, choices=["moco3", "simclr"])
    p.add_argument("--model", default=DEFAULT_MODEL, choices=["ViViT", "X3D", "C3D"])
    p.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    p.add_argument("--input-csv", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--figsize-scale", type=float, default=0.45)
    return p.parse_args()


def experiment_tag(experiment: str, model: str, epoch: int) -> str:
    return f"{experiment}_{model}_{epoch}"


def cramers_v_from_table(table: pd.DataFrame) -> float:
    chi2, _, _, _ = chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    if n == 0:
        return float("nan")
    r, k = table.shape
    denom = n * max(min(r - 1, k - 1), 1)
    return (chi2 / denom) ** 0.5


def strength_label(v: float) -> str:
    if pd.isna(v):
        return "nan"
    if v < 0.1:
        return "very_weak"
    if v < 0.3:
        return "weak"
    if v < 0.5:
        return "moderate"
    return "strong"


def save_heatmap(df: pd.DataFrame, out_path: Path, title: str, figsize_scale: float) -> None:
    if df.empty:
        return

    width = max(8, df.shape[1] * figsize_scale)
    height = max(4, df.shape[0] * 0.6)

    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(df.to_numpy(), aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(df.columns.name if df.columns.name else "code")
    ax.set_ylabel(df.index.name if df.index.name else "cluster")
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels([str(c) for c in df.columns], rotation=90, fontsize=8)
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels([str(i) for i in df.index], fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_barplot(rate_table: pd.DataFrame, out_path: Path, title: str) -> None:
    if rate_table.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    melted = rate_table.reset_index().melt(id_vars="cluster", var_name="code", value_name="rate")

    for cluster_id, sub in melted.groupby("cluster"):
        ax.plot(sub["code"].astype(str), sub["rate"], marker="o", linewidth=1, label=f"cluster {cluster_id}")

    ax.set_title(title)
    ax.set_xlabel("code")
    ax.set_ylabel("rate")
    ax.tick_params(axis="x", rotation=90)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    tag = experiment_tag(args.experiment, args.model, args.epoch)

    input_csv = Path(args.input_csv) if args.input_csv else BASE_COARSE_OUTPUT_DIR / tag / "merged_analysis_table_coarse.csv"
    out_root = Path(args.output_dir) if args.output_dir else BASE_COARSE_OUTPUT_DIR / tag / "first_order_coarse"

    print("=" * 80)
    print("[1/4] Loading merged_analysis_table_coarse.csv")
    print(f"Input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded rows: {len(df)}")
    print(f"Clusters: {sorted(df['cluster'].dropna().unique().tolist())}")

    print("=" * 80)
    print("[2/4] Saving cluster/feature summaries")
    cluster_counts = df["cluster"].value_counts().sort_index().rename_axis("cluster").reset_index(name="n_rows")
    cluster_counts["rate"] = cluster_counts["n_rows"] / len(df)
    print(cluster_counts.to_string(index=False))

    summary_rows = []
    all_features = sorted(set(FEATURE_SETS["v1"] + FEATURE_SETS["v2"]))
    for feat in all_features:
        ser = pd.to_numeric(df[feat], errors="coerce")
        summary_rows.append({
            "feature": feat,
            "non_null_rows": int(ser.notna().sum()),
            "unique_codes": int(ser.dropna().astype(int).nunique()),
            "min_code": int(ser.dropna().astype(int).min()) if ser.notna().any() else None,
            "max_code": int(ser.dropna().astype(int).max()) if ser.notna().any() else None,
            "neg1_rows": int((ser == -1).sum()),
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    print("=" * 80)
    print("[3/4] Building first-order heatmaps and statistics")
    stat_frames = []

    for version, features in FEATURE_SETS.items():
        heat_dir = out_root / version / "heatmaps"
        bar_dir = out_root / version / "bars"
        table_dir = out_root / version / "tables"
        stats_dir = out_root / version / "stats"

        for d in [heat_dir, bar_dir, table_dir, stats_dir]:
            d.mkdir(parents=True, exist_ok=True)

        cluster_counts.to_csv(table_dir / "cluster_counts.csv", index=False, encoding="utf-8-sig")
        summary_df.to_csv(table_dir / "feature_code_summary.csv", index=False, encoding="utf-8-sig")

        for feat in features:
            sub = df[["cluster", feat]].copy()
            sub = sub.dropna()
            sub[feat] = pd.to_numeric(sub[feat], errors="coerce")
            sub = sub.dropna()
            sub[feat] = sub[feat].astype(int)

            count_table = pd.crosstab(sub["cluster"], sub[feat])
            count_table.index.name = "cluster"
            count_table.columns.name = feat
            rate_table = count_table.div(count_table.sum(axis=1), axis=0).fillna(0.0)

            count_table.to_csv(table_dir / f"{feat}_count_table.csv", encoding="utf-8-sig")
            rate_table.to_csv(table_dir / f"{feat}_rate_table.csv", encoding="utf-8-sig")

            save_heatmap(
                count_table,
                heat_dir / f"{feat}__count_heatmap.png",
                f"{version} | {feat} | count",
                args.figsize_scale,
            )
            save_heatmap(
                rate_table,
                heat_dir / f"{feat}__rate_heatmap.png",
                f"{version} | {feat} | rate",
                args.figsize_scale,
            )
            save_barplot(
                rate_table,
                bar_dir / f"{feat}__rate_barplot.png",
                f"{version} | {feat} | rate by cluster",
            )

            _, p_value, _, _ = chi2_contingency(count_table, correction=False)
            v = cramers_v_from_table(count_table)

            stat_frames.append({
                "version": version,
                "feature": feat,
                "n_rows_for_test": int(count_table.shape[0]),
                "n_cols_for_test": int(count_table.shape[1]),
                "n_total_samples": int(count_table.to_numpy().sum()),
                "n_clusters_present": int(count_table.shape[0]),
                "n_codes_present": int(count_table.shape[1]),
                "p_value": float(p_value),
                "cramers_v": float(v),
                "strength": strength_label(float(v)),
            })

            print(
                f"Done [{version}]: {feat} | "
                f"clusters={count_table.shape[0]} | "
                f"codes={count_table.shape[1]} | "
                f"p={p_value:.3e} | V={v:.4f}"
            )

        version_stats = pd.DataFrame([r for r in stat_frames if r["version"] == version]).sort_values(
            ["cramers_v", "p_value"], ascending=[False, True]
        )
        version_stats.to_csv(stats_dir / "chi_square_cramers_v.csv", index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("[4/4] Saving stats summary")
    all_stats = pd.DataFrame(stat_frames).sort_values(["version", "cramers_v"], ascending=[True, False])
    print(all_stats.to_string(index=False))

    out_root.mkdir(parents=True, exist_ok=True)
    all_stats.to_csv(out_root / "chi_square_cramers_v_all_versions.csv", index=False, encoding="utf-8-sig")
    print(f"Saved outputs under: {out_root}")
    print("Done.")


if __name__ == "__main__":
    main()