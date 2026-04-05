import pandas as pd
import umap
import matplotlib.pyplot as plt


def umap_visualization(
    csv_path,
    output_path,
    n_neighbors=100,
    min_dist=0.01,
    random_state=42
):
    """
    csv_path에서 numeric feature column들을 자동 추출해서 2D UMAP 시각화 저장
    """
    df = pd.read_csv(csv_path)

    exclude_cols = {"cluster", "data"}
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    if len(feature_cols) == 0:
        raise ValueError(f"No feature columns found in {csv_path}")

    vectors = df[feature_cols].values

    n_neighbors = min(n_neighbors, max(2, len(df) - 1))

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        metric="cosine",
        min_dist=min_dist,
        random_state=random_state
    )
    reduced_vectors = reducer.fit_transform(vectors)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    unique_clusters = sorted(df["cluster"].unique())
    num_clusters = len(unique_clusters)
    cmap = plt.get_cmap("tab10") if num_clusters <= 10 else plt.get_cmap("tab20")

    for idx, cluster_id in enumerate(unique_clusters):
        cluster_mask = df["cluster"] == cluster_id
        cluster_points = reduced_vectors[cluster_mask]
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=f"Cluster {cluster_id}",
            color=cmap(idx % cmap.N),
            s=5
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.title("2D UMAP projection")
    ax.legend(title="Cluster", loc="best")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"[UMAP] Saved: {output_path}")