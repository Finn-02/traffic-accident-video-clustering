import os
import shutil
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


def extract_video(
    video_dir,
    saved_result_csv,
    extracted_video_dir,
    top_k=5,
    bottom_k=20,
):
    video_dir = Path(video_dir)
    saved_result_csv = Path(saved_result_csv)
    extracted_video_dir = Path(extracted_video_dir)

    if not saved_result_csv.exists():
        raise FileNotFoundError(f"Cluster result CSV not found: {saved_result_csv}")

    if not video_dir.exists():
        raise FileNotFoundError(f"Original video directory not found: {video_dir}")

    result_df = pd.read_csv(saved_result_csv)

    required_cols = {"cluster", "data"}
    if not required_cols.issubset(result_df.columns):
        raise ValueError(
            f"CSV must contain columns {required_cols}, "
            f"but got columns: {list(result_df.columns)}"
        )

    # feature column들만 추출
    feature_cols = [col for col in result_df.columns if col not in ["cluster", "data"]]
    if len(feature_cols) == 0:
        raise ValueError("No feature vector columns found in CSV.")

    vectors = result_df[feature_cols].to_numpy(dtype=np.float32)
    vectors = normalize(vectors, axis=1)

    cluster_labels = sorted(result_df["cluster"].unique())

    # 기존 결과 폴더 삭제 후 재생성
    if extracted_video_dir.exists():
        shutil.rmtree(extracted_video_dir)
    extracted_video_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[Start Extract Video]")
    print(f"VIDEO_DIR           : {video_dir}")
    print(f"SAVED_RESULT_CSV    : {saved_result_csv}")
    print(f"EXTRACTED_VIDEO_DIR : {extracted_video_dir}")
    print(f"TOP_K               : {top_k}")
    print(f"BOTTOM_K            : {bottom_k}")
    print(f"NUM_SAMPLES         : {len(result_df)}")
    print(f"NUM_CLUSTERS        : {len(cluster_labels)}")
    print("=" * 80)

    for cluster_idx in cluster_labels:
        cluster_mask = result_df["cluster"] == cluster_idx
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) == 0:
            print(f"[Warning] Cluster {cluster_idx} is empty. Skipping.")
            continue

        cluster_vectors = vectors[cluster_indices]

        # cluster centroid
        centroid = cluster_vectors.mean(axis=0, keepdims=True)
        centroid = normalize(centroid, axis=1)

        # cosine similarity
        sims = (cluster_vectors @ centroid.T).squeeze(1)

        # top / bottom index 선택
        sorted_desc_local = np.argsort(sims)[::-1]
        sorted_asc_local = np.argsort(sims)

        top_local_idx = sorted_desc_local[: min(top_k, len(sorted_desc_local))]
        bottom_local_idx = sorted_asc_local[: min(bottom_k, len(sorted_asc_local))]

        top_global_idx = cluster_indices[top_local_idx]
        bottom_global_idx = cluster_indices[bottom_local_idx]

        # 폴더 구조
        cluster_folder = extracted_video_dir / f"cluster_{cluster_idx}"
        top_folder = cluster_folder / f"top_{top_k}"
        bottom_folder = cluster_folder / f"bottom_{bottom_k}"

        top_folder.mkdir(parents=True, exist_ok=True)
        bottom_folder.mkdir(parents=True, exist_ok=True)

        print(
            f"[Cluster {cluster_idx}] total={len(cluster_indices)} | "
            f"top={len(top_global_idx)} | bottom={len(bottom_global_idx)}"
        )

        # top 저장
        for rank, video_idx in enumerate(top_global_idx, start=1):
            rel_path = str(result_df.iloc[video_idx]["data"])
            src_path = video_dir / rel_path

            if not src_path.exists():
                print(f"  [Top Missing] {src_path}")
                continue

            # 파일명 충돌 방지용 prefix
            src_name = Path(rel_path).name
            dst_name = f"rank{rank:02d}_{src_name}"
            dst_path = top_folder / dst_name

            shutil.copy2(src_path, dst_path)
            print(f"  [Top {rank}] copied -> {dst_path}")

        # bottom 저장
        for rank, video_idx in enumerate(bottom_global_idx, start=1):
            rel_path = str(result_df.iloc[video_idx]["data"])
            src_path = video_dir / rel_path

            if not src_path.exists():
                print(f"  [Bottom Missing] {src_path}")
                continue

            src_name = Path(rel_path).name
            dst_name = f"rank{rank:02d}_{src_name}"
            dst_path = bottom_folder / dst_name

            shutil.copy2(src_path, dst_path)
            print(f"  [Bottom {rank}] copied -> {dst_path}")

    print("=" * 80)
    print("[Extract Complete]")
    print(f"Saved to: {extracted_video_dir}")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video_dir",
        type=str,
        default="/home/ellenhong/datasets/video_original",
        help="Directory containing original videos."
    )
    parser.add_argument(
        "--saved_result_csv",
        type=str,
        default="/home/ellenhong/VideoContrastive/moco3/experiments_32/moco3_ViViT_100/moco3_ViViT_cluster_result.csv",
        help="Cluster result CSV generated by train_DDP.py"
    )
    parser.add_argument(
        "--extracted_video_dir",
        type=str,
        default="/home/ellenhong/VideoContrastive/moco3/experiments_32/moco3_ViViT_100/moco3_ViViT_extracted_video",
        help="Output directory for extracted representative/failure videos."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of representative top videos per cluster."
    )
    parser.add_argument(
        "--bottom_k",
        type=int,
        default=20,
        help="Number of bottom videos per cluster for failure analysis."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    extract_video(
        video_dir=args.video_dir,
        saved_result_csv=args.saved_result_csv,
        extracted_video_dir=args.extracted_video_dir,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
    )