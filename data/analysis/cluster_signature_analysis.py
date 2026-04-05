#!/usr/bin/env python3
import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run cluster signature analysis in one pass: "
            "first-order standardized residuals + second-order one-vs-rest pair enrichment."
        )
    )
    p.add_argument("--experiment", default=DEFAULT_EXPERIMENT, choices=["moco3", "simclr"])
    p.add_argument("--model", default=DEFAULT_MODEL, choices=["ViViT", "X3D", "C3D"])
    p.add_argument("--epoch", type=int, default=DEFAULT_EPOCH)
    p.add_argument("--input-csv", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--common-threshold",
        type=float,
        default=0.75,
        help="If a code/pair appears in top-k of at least this fraction of clusters, mark it as globally shared.",
    )
    return p.parse_args()


def experiment_tag(experiment: str, model: str, epoch: int) -> str:
    return f"{experiment}_{model}_{epoch}"


def standardized_residuals(contingency: pd.DataFrame) -> pd.DataFrame:
    observed = contingency.to_numpy(dtype=float)
    total = observed.sum()
    if total == 0:
        return pd.DataFrame(index=contingency.index, columns=contingency.columns, data=np.nan)

    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / total

    row_prop = row_sum / total
    col_prop = col_sum / total
    denom = np.sqrt(expected * (1 - row_prop) * (1 - col_prop))
    denom[denom == 0] = np.nan
    resid = (observed - expected) / denom

    return pd.DataFrame(resid, index=contingency.index, columns=contingency.columns)


def first_order_signature_tables(
    df: pd.DataFrame,
    features: List[str],
    top_k: int,
    common_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cluster_ids = sorted(df["cluster"].dropna().astype(int).unique().tolist())
    sig_rows = []
    shared_rows = []
    summary_rows = []

    for feature in features:
        sub = df[["cluster", feature]].dropna().copy()
        sub[feature] = pd.to_numeric(sub[feature], errors="coerce")
        sub = sub.dropna()
        sub[feature] = sub[feature].astype(int)
        if sub.empty:
            continue

        contingency = pd.crosstab(sub["cluster"], sub[feature])
        residual_df = standardized_residuals(contingency)
        rate_df = contingency.div(contingency.sum(axis=1), axis=0).fillna(0.0)
        overall_rate = contingency.sum(axis=0) / contingency.to_numpy().sum()

        feature_top = []
        for cluster_id in cluster_ids:
            if cluster_id not in residual_df.index:
                continue
            cluster_series = residual_df.loc[cluster_id].sort_values(ascending=False)
            top_codes = cluster_series.head(top_k)
            for code, residual in top_codes.items():
                c_rate = float(rate_df.loc[cluster_id, code])
                g_rate = float(overall_rate.loc[code])
                feature_top.append((cluster_id, int(code), float(residual), c_rate, g_rate))

        # globally shared codes: appear in top-k for many clusters
        count_top = pd.Series([x[1] for x in feature_top]).value_counts()
        min_clusters = max(1, int(np.ceil(common_threshold * len(cluster_ids))))
        shared_codes = set(count_top[count_top >= min_clusters].index.tolist())

        for cluster_id, code, residual, c_rate, g_rate in feature_top:
            row = {
                "feature": feature,
                "cluster": int(cluster_id),
                "code": int(code),
                "standardized_residual": float(residual),
                "cluster_rate": c_rate,
                "overall_rate": g_rate,
                "is_globally_shared": int(code in shared_codes),
            }
            sig_rows.append(row)
            if code in shared_codes:
                shared_rows.append(row)

        summary_rows.append({
            "feature": feature,
            "n_codes": int(contingency.shape[1]),
            "n_clusters": int(contingency.shape[0]),
            "n_shared_codes": int(len(shared_codes)),
            "shared_codes": ", ".join(map(str, sorted(shared_codes))) if shared_codes else "",
        })

    sig_df = pd.DataFrame(sig_rows)
    shared_df = pd.DataFrame(shared_rows)
    if not sig_df.empty:
        distinctive_df = sig_df[sig_df["is_globally_shared"] == 0].copy()
        distinctive_df = distinctive_df.sort_values(["feature", "cluster", "standardized_residual"], ascending=[True, True, False])
    else:
        distinctive_df = pd.DataFrame(columns=[
            "feature", "cluster", "code", "standardized_residual", "cluster_rate", "overall_rate", "is_globally_shared"
        ])
    return sig_df, shared_df, pd.DataFrame(summary_rows), distinctive_df


def save_first_order_heatmaps(
    df: pd.DataFrame,
    features: List[str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for feature in features:
        sub = df[["cluster", feature]].dropna().copy()
        sub[feature] = pd.to_numeric(sub[feature], errors="coerce")
        sub = sub.dropna()
        sub[feature] = sub[feature].astype(int)
        if sub.empty:
            continue
        contingency = pd.crosstab(sub["cluster"], sub[feature])
        resid_df = standardized_residuals(contingency)

        fig_w = max(8, 0.45 * resid_df.shape[1])
        fig_h = max(4, 0.7 * resid_df.shape[0])
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(resid_df.to_numpy(), aspect="auto")
        ax.set_title(f"{feature} | standardized residuals")
        ax.set_xlabel(feature)
        ax.set_ylabel("cluster")
        ax.set_xticks(range(resid_df.shape[1]))
        ax.set_xticklabels([str(c) for c in resid_df.columns], rotation=90, fontsize=8)
        ax.set_yticks(range(resid_df.shape[0]))
        ax.set_yticklabels([str(i) for i in resid_df.index], fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        plt.tight_layout()
        fig.savefig(out_dir / f"{feature}__standardized_residual_heatmap.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def second_order_pair_enrichment(
    df: pd.DataFrame,
    features: List[str],
    top_k: int,
    common_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cluster_ids = sorted(df["cluster"].dropna().astype(int).unique().tolist())
    all_rows = []
    shared_rows = []
    summary_rows = []

    for f1, f2 in combinations(features, 2):
        sub = df[["cluster", f1, f2]].dropna().copy()
        sub[f1] = pd.to_numeric(sub[f1], errors="coerce")
        sub[f2] = pd.to_numeric(sub[f2], errors="coerce")
        sub = sub.dropna()
        sub[f1] = sub[f1].astype(int)
        sub[f2] = sub[f2].astype(int)
        if sub.empty:
            continue

        sub["pair_code"] = sub[f1].astype(str) + "|" + sub[f2].astype(str)
        total_pair_counts = sub["pair_code"].value_counts()
        total_pair_rate = total_pair_counts / len(sub)

        pair_name = f"{f1}__{f2}"
        pair_top = []
        for cluster_id in cluster_ids:
            pos = sub[sub["cluster"] == cluster_id]
            neg = sub[sub["cluster"] != cluster_id]
            if pos.empty or neg.empty:
                continue

            pos_counts = pos["pair_code"].value_counts()
            neg_counts = neg["pair_code"].value_counts()
            pos_rate = pos_counts / len(pos)
            neg_rate = neg_counts / len(neg)

            candidate_pairs = sorted(set(pos_rate.index.tolist()) | set(neg_rate.index.tolist()))
            rows = []
            for pair_code in candidate_pairs:
                pr = float(pos_rate.get(pair_code, 0.0))
                nr = float(neg_rate.get(pair_code, 0.0))
                lift = (pr + 1e-9) / (nr + 1e-9)
                log2_lift = float(np.log2(lift))
                rows.append((pair_code, log2_lift, pr, nr, float(total_pair_rate.get(pair_code, 0.0))))

            rows = sorted(rows, key=lambda x: x[1], reverse=True)[:top_k]
            for pair_code, log2_lift, pr, nr, tr in rows:
                c1, c2 = pair_code.split("|")
                rec = {
                    "pair_name": pair_name,
                    "feature_1": f1,
                    "feature_2": f2,
                    "cluster": int(cluster_id),
                    "code_1": int(c1),
                    "code_2": int(c2),
                    "pair_code": pair_code,
                    "log2_lift_one_vs_rest": float(log2_lift),
                    "cluster_rate": float(pr),
                    "rest_rate": float(nr),
                    "overall_rate": float(tr),
                }
                all_rows.append(rec)
                pair_top.append((cluster_id, pair_code))

        count_top = pd.Series([x[1] for x in pair_top]).value_counts()
        min_clusters = max(1, int(np.ceil(common_threshold * len(cluster_ids))))
        shared_pairs = set(count_top[count_top >= min_clusters].index.tolist())

        summary_rows.append({
            "pair_name": pair_name,
            "feature_1": f1,
            "feature_2": f2,
            "n_shared_pairs": int(len(shared_pairs)),
            "shared_pairs": ", ".join(sorted(shared_pairs)) if shared_pairs else "",
        })

        for row in all_rows:
            if row["pair_name"] == pair_name and row["pair_code"] in shared_pairs:
                rr = row.copy()
                rr["is_globally_shared"] = 1
                shared_rows.append(rr)

    all_df = pd.DataFrame(all_rows)
    if all_df.empty:
        return all_df, pd.DataFrame(), pd.DataFrame(summary_rows), pd.DataFrame()

    all_df["is_globally_shared"] = 0
    shared_lookup = {(r["pair_name"], r["pair_code"]) for r in shared_rows}
    all_df.loc[all_df.apply(lambda x: (x["pair_name"], x["pair_code"]) in shared_lookup, axis=1), "is_globally_shared"] = 1
    distinctive_df = all_df[all_df["is_globally_shared"] == 0].copy()
    distinctive_df = distinctive_df.sort_values(["pair_name", "cluster", "log2_lift_one_vs_rest"], ascending=[True, True, False])
    shared_df = all_df[all_df["is_globally_shared"] == 1].copy()
    return all_df, shared_df, pd.DataFrame(summary_rows), distinctive_df


def save_second_order_cluster_tables(
    second_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if second_df.empty:
        return
    for pair_name, pair_sub in second_df.groupby("pair_name"):
        pair_dir = out_dir / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)
        for cluster_id, cluster_sub in pair_sub.groupby("cluster"):
            cluster_sub = cluster_sub.sort_values("log2_lift_one_vs_rest", ascending=False)
            cluster_sub.to_csv(pair_dir / f"cluster_{int(cluster_id)}_pair_signature_table.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    tag = experiment_tag(args.experiment, args.model, args.epoch)

    input_csv = Path(args.input_csv) if args.input_csv else BASE_COARSE_OUTPUT_DIR / tag / "merged_analysis_table_coarse.csv"
    output_root = Path(args.output_dir) if args.output_dir else BASE_COARSE_OUTPUT_DIR / tag / "cluster_signature_analysis"
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[1/5] Loading merged_analysis_table_coarse.csv")
    print(f"Input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded rows: {len(df)}")
    print(f"Clusters: {sorted(df['cluster'].dropna().astype(int).unique().tolist())}")

    for version, features in FEATURE_SETS.items():
        print("=" * 80)
        print(f"[2/5] Running first-order signatures ({version})")
        version_root = output_root / version
        first_root = version_root / "first_order_signatures"
        second_root = version_root / "second_order_signatures"
        first_root.mkdir(parents=True, exist_ok=True)
        second_root.mkdir(parents=True, exist_ok=True)

        sig_df, shared_df, summary_df, distinctive_df = first_order_signature_tables(
            df=df,
            features=features,
            top_k=args.top_k,
            common_threshold=args.common_threshold,
        )
        sig_df.to_csv(first_root / "all_topk_standardized_residuals.csv", index=False, encoding="utf-8-sig")
        shared_df.to_csv(first_root / "globally_shared_first_order_codes.csv", index=False, encoding="utf-8-sig")
        summary_df.to_csv(first_root / "first_order_shared_code_summary.csv", index=False, encoding="utf-8-sig")
        distinctive_df.to_csv(first_root / "cluster_distinctive_first_order_codes.csv", index=False, encoding="utf-8-sig")
        save_first_order_heatmaps(df, features, first_root / "residual_heatmaps")

        if not distinctive_df.empty:
            preview = distinctive_df.groupby(["feature", "cluster"]).head(min(args.top_k, 3))
            print(preview.to_string(index=False))

        print("=" * 80)
        print(f"[3/5] Running second-order signatures ({version})")
        all_pair_df, shared_pair_df, pair_summary_df, distinctive_pair_df = second_order_pair_enrichment(
            df=df,
            features=features,
            top_k=args.top_k,
            common_threshold=args.common_threshold,
        )
        all_pair_df.to_csv(second_root / "all_topk_pair_log2_lift.csv", index=False, encoding="utf-8-sig")
        shared_pair_df.to_csv(second_root / "globally_shared_second_order_pairs.csv", index=False, encoding="utf-8-sig")
        pair_summary_df.to_csv(second_root / "second_order_shared_pair_summary.csv", index=False, encoding="utf-8-sig")
        distinctive_pair_df.to_csv(second_root / "cluster_distinctive_second_order_pairs.csv", index=False, encoding="utf-8-sig")
        save_second_order_cluster_tables(distinctive_pair_df, second_root / "pair_signature_tables")

        if not distinctive_pair_df.empty:
            preview2 = distinctive_pair_df.groupby(["pair_name", "cluster"]).head(min(args.top_k, 3))
            print(preview2.to_string(index=False))

        print("=" * 80)
        print(f"[4/5] Saving version summary ({version})")
        version_summary = pd.DataFrame([
            {
                "version": version,
                "n_first_order_rows": int(len(sig_df)),
                "n_first_order_distinctive_rows": int(len(distinctive_df)),
                "n_first_order_shared_rows": int(len(shared_df)),
                "n_second_order_rows": int(len(all_pair_df)),
                "n_second_order_distinctive_rows": int(len(distinctive_pair_df)),
                "n_second_order_shared_rows": int(len(shared_pair_df)),
            }
        ])
        version_summary.to_csv(version_root / "signature_analysis_summary.csv", index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("[5/5] Done")
    print(f"Saved outputs under: {output_root}")


if __name__ == "__main__":
    main()
