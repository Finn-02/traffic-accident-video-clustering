#!/usr/bin/env python3
import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


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
    p = argparse.ArgumentParser(description="Second-order coarse analysis.")
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT, choices=["moco3", "simclr"])
    p.add_argument("--model", default=DEFAULT_MODEL, choices=["ViViT", "X3D", "C3D"])
    p.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    p.add_argument("--input-csv", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


def experiment_tag(experiment: str, model: str, epoch: int) -> str:
    return f"{experiment}_{model}_{epoch}"


def save_heatmap(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df.empty:
        return

    width = max(6, df.shape[1] * 0.7)
    height = max(5, df.shape[0] * 0.55)

    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(df.to_numpy(), aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(df.columns.name if df.columns.name else "feature_j")
    ax.set_ylabel(df.index.name if df.index.name else "feature_i")
    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels([str(c) for c in df.columns], rotation=90, fontsize=8)
    ax.set_yticks(range(df.shape[0]))
    ax.set_yticklabels([str(i) for i in df.index], fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_contact_sheet(image_paths: List[Path], out_path: Path, title: str, ncols: int = 4) -> None:
    valid = [p for p in image_paths if p.exists()]
    if not valid:
        return

    import matplotlib.image as mpimg
    import numpy as np

    n = len(valid)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, img_path in zip(axes, valid):
        ax.imshow(mpimg.imread(img_path))
        ax.set_title(img_path.stem, fontsize=8)
        ax.axis("off")

    for ax in axes[len(valid):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def pair_assoc_score(df: pd.DataFrame, cluster_col: str, f1: str, f2: str) -> float:
    pair_counts = df.groupby([cluster_col, f1, f2]).size().reset_index(name="n")
    pair_counts["cluster_total"] = pair_counts.groupby(cluster_col)["n"].transform("sum")
    pair_counts["rate_in_cluster"] = pair_counts["n"] / pair_counts["cluster_total"]
    max_per_cluster = pair_counts.groupby(cluster_col)["rate_in_cluster"].max()
    return float(max_per_cluster.mean())


def main():
    args = parse_args()
    tag = experiment_tag(args.experiment, args.model, args.epoch)

    input_csv = Path(args.input_csv) if args.input_csv else BASE_COARSE_OUTPUT_DIR / tag / "merged_analysis_table_coarse.csv"
    out_root = Path(args.output_dir) if args.output_dir else BASE_COARSE_OUTPUT_DIR / tag / "second_order_coarse"

    print("=" * 80)
    print("[1/4] Loading merged_analysis_table_coarse.csv")
    print(f"Input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded rows: {len(df)}")
    clusters = sorted(df["cluster"].dropna().astype(int).unique().tolist())
    print(f"Clusters: {clusters}")

    print("=" * 80)
    print("[2/4] Building second-order outputs")
    pair_stat_rows = []

    for version, features in FEATURE_SETS.items():
        print(f"[INFO] Version: {version} | features={features}")

        version_root = out_root / version
        heat_root = version_root / "joint_heatmaps"
        topk_root = version_root / "joint_topk_tables"
        summary_root = version_root / "cluster_summaries"
        stats_root = version_root / "stats"

        for d in [heat_root, topk_root, summary_root, stats_root]:
            d.mkdir(parents=True, exist_ok=True)

        cluster_to_images: Dict[int, List[Path]] = {c: [] for c in clusters}

        for f1, f2 in combinations(features, 2):
            pair_name = f"{f1}__{f2}"
            pair_heat_dir = heat_root / pair_name
            pair_topk_dir = topk_root / pair_name
            pair_heat_dir.mkdir(parents=True, exist_ok=True)
            pair_topk_dir.mkdir(parents=True, exist_ok=True)

            score = pair_assoc_score(df.dropna(subset=["cluster", f1, f2]), "cluster", f1, f2)
            pair_stat_rows.append({
                "version": version,
                "feature_1": f1,
                "feature_2": f2,
                "pair_name": pair_name,
                "mean_max_rate_across_clusters": score,
            })

            pair_imgs = []

            for cluster_id in clusters:
                sub = df[df["cluster"] == cluster_id][[f1, f2]].copy()
                sub = sub.dropna()
                sub[f1] = pd.to_numeric(sub[f1], errors="coerce")
                sub[f2] = pd.to_numeric(sub[f2], errors="coerce")
                sub = sub.dropna()
                sub[f1] = sub[f1].astype(int)
                sub[f2] = sub[f2].astype(int)

                if sub.empty:
                    continue

                count_table = pd.crosstab(sub[f1], sub[f2])
                count_table.index.name = f1
                count_table.columns.name = f2
                rate_table = count_table / count_table.to_numpy().sum()

                count_table.to_csv(pair_topk_dir / f"cluster_{cluster_id}__count_table.csv", encoding="utf-8-sig")
                rate_table.to_csv(pair_topk_dir / f"cluster_{cluster_id}__rate_table.csv", encoding="utf-8-sig")

                count_img = pair_heat_dir / f"cluster_{cluster_id}__count.png"
                rate_img = pair_heat_dir / f"cluster_{cluster_id}__rate.png"

                save_heatmap(
                    count_table,
                    count_img,
                    f"{version} | cluster {cluster_id} | {f1} x {f2} | count",
                )
                save_heatmap(
                    rate_table,
                    rate_img,
                    f"{version} | cluster {cluster_id} | {f1} x {f2} | rate",
                )

                pair_imgs.append(rate_img)
                cluster_to_images[cluster_id].append(rate_img)

                topk = (
                    sub.groupby([f1, f2]).size().reset_index(name="count")
                    .sort_values("count", ascending=False)
                    .reset_index(drop=True)
                )
                topk["rate"] = topk["count"] / topk["count"].sum()
                topk.insert(0, "rank", range(1, len(topk) + 1))
                topk.head(args.top_k).to_csv(
                    pair_topk_dir / f"cluster_{cluster_id}_top{args.top_k}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

            make_contact_sheet(
                pair_imgs,
                pair_heat_dir / f"{pair_name}__cluster_contact_sheet.png",
                f"{version} | {pair_name} | cluster contact sheet",
                ncols=4,
            )
            print(f"Done [{version}]: {pair_name} | clusters={len(pair_imgs)}")

        for cluster_id, img_list in cluster_to_images.items():
            make_contact_sheet(
                img_list,
                summary_root / f"cluster_{cluster_id}_summary.png",
                f"{version} | cluster {cluster_id} summary",
                ncols=3,
            )

    print("=" * 80)
    print("[3/4] Saving pair-level stats")
    stats_df = pd.DataFrame(pair_stat_rows).sort_values(
        ["version", "mean_max_rate_across_clusters"],
        ascending=[True, False],
    )
    print(stats_df.to_string(index=False))

    out_root.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(
        out_root / "pair_level_cluster_association_all_versions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("=" * 80)
    print("[4/4] Done")
    print(f"Saved outputs under: {out_root}")


if __name__ == "__main__":
    main()