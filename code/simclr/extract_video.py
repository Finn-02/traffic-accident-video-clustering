import os
import shutil
import numpy as np
import pandas as pd
from sklearn import preprocessing


def extract_video(
    video_dir,
    saved_result_csv,
    extracted_video_dir,
    top_k=5
):
    """
    cluster_result.csv를 읽어서 각 cluster별 대표 영상 top_k개를 복사
    """
    if not os.path.exists(saved_result_csv):
        print(f"[extract_video] Skip: cluster result not found -> {saved_result_csv}")
        return

    if not os.path.isdir(video_dir):
        print(f"[extract_video] Skip: video_dir not found -> {video_dir}")
        return

    result_df = pd.read_csv(saved_result_csv)

    if "cluster" not in result_df.columns or "data" not in result_df.columns:
        raise ValueError("cluster_result.csv must contain 'cluster' and 'data' columns.")

    feature_cols = [col for col in result_df.columns if col not in ["cluster", "data"]]
    if len(feature_cols) == 0:
        raise ValueError("No feature columns found in cluster_result.csv")

    if os.path.isdir(extracted_video_dir):
        shutil.rmtree(extracted_video_dir)
    os.makedirs(extracted_video_dir, exist_ok=True)

    vectors = result_df[feature_cols].values
    vectors = preprocessing.normalize(vectors)

    cluster_ids = sorted(result_df["cluster"].unique())

    for cluster_id in cluster_ids:
        cluster_mask = result_df["cluster"] == cluster_id
        cluster_vectors = vectors[cluster_mask]
        cluster_rows = result_df[cluster_mask].reset_index(drop=True)

        cluster_folder = os.path.join(extracted_video_dir, str(cluster_id))
        os.makedirs(cluster_folder, exist_ok=True)

        if len(cluster_vectors) == 0:
            continue

        centroid = cluster_vectors.mean(axis=0, keepdims=True)
        centroid = preprocessing.normalize(centroid)
        sim_scores = (cluster_vectors @ centroid.T).reshape(-1)

        top_indexes = np.argsort(sim_scores)[::-1][:top_k]

        for video_idx in top_indexes:
            video_name = cluster_rows.iloc[video_idx]["data"]
            src_path = os.path.join(video_dir, video_name)
            dst_path = os.path.join(cluster_folder, video_name)

            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)

    print(f"[extract_video] Saved representative videos to: {extracted_video_dir}")


if __name__ == "__main__":
    # 예시
    extract_video(
        video_dir="../../../Datasets/VideoClustering/video_original",
        saved_result_csv="./cluster_result.csv",
        extracted_video_dir="./extracted_video",
        top_k=5
    )