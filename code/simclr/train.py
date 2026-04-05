import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import models
from losses import nt_xent_loss
from dataloader import VideoDataset, build_collate_fn
from extract_video import extract_video
from umap_visualization import umap_visualization


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, local_rank, world_size, device


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_prefix(framework_name: str, model_name: str) -> str:
    return f"{framework_name}_{model_name}"


def get_paths(save_dir: str, framework_name: str, model_name: str):
    prefix = get_prefix(framework_name, model_name)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    return {
        "save_dir": save_dir,
        "prefix": prefix,
        "loss_png": save_dir / f"{prefix}_Training_Loss.png",
        "ckpt": save_dir / f"{prefix}_ckpt.weights.pth",
        "cluster_csv": save_dir / f"{prefix}_cluster_result.csv",
        "centroid_csv": save_dir / f"{prefix}_cluster_centers.csv",
        "sil_png": save_dir / f"{prefix}_silhouette_score.png",
        "umap_png": save_dir / f"{prefix}_umap_projection_2d.png",
        "config_json": save_dir / f"{prefix}_run_config.json",
        "extracted_video_dir": save_dir / f"{prefix}_extracted_video",
    }


def save_config_if_rank0(rank: int, path_dict: dict, run_args):
    if rank != 0:
        return
    with open(path_dict["config_json"], "w", encoding="utf-8") as f:
        json.dump(vars(run_args), f, indent=2, ensure_ascii=False)


def get_model(param_img_size, param_frame, device, model_name, output_dim):
    model = models.build_model(
        model_name=model_name,
        image_size=param_img_size,
        num_frames=param_frame,
        output_dim=output_dim,
    ).to(device)

    params_m = models.count_trainable_params_m(model)
    print(f"[Model] {model_name} | Trainable Parameters: {params_m:.3f}M")
    return model


def train(
    param_frame,
    param_img_size,
    dataset_dir,
    epochs,
    batch_size,
    rank,
    world_size,
    device,
    model_name,
    framework_name,
    save_dir,
    num_workers,
    output_dim,
    run_args,
):
    if batch_size % world_size != 0:
        raise ValueError(f"batch_size({batch_size}) must be divisible by world_size({world_size})")

    path_dict = get_paths(save_dir, framework_name, model_name)
    save_config_if_rank0(rank, path_dict, run_args)

    local_batch_size = batch_size // world_size

    if rank == 0:
        print("=" * 80)
        print(f"[Start Train] framework={framework_name} | model={model_name}")
        print(f"[Device] {device}")
        print(f"[GPU count] {torch.cuda.device_count()}")
        print(f"[World size] {world_size}")
        print(f"[Global batch size] {batch_size}")
        print(f"[Local batch size] {local_batch_size}")
        print(f"[Dataset dir] {dataset_dir}")
        print(f"[Save dir] {path_dict['save_dir']}")
        print("=" * 80)

    dataset = VideoDataset(data_dir=dataset_dir, frame=param_frame, img_size=param_img_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    collate_fn = build_collate_fn(img_size=param_img_size)

    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": local_batch_size,
        "sampler": sampler,
        "collate_fn": collate_fn,
        "drop_last": True,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 4
        dataloader_kwargs["persistent_workers"] = True

    data_loader = DataLoader(**dataloader_kwargs)

    model = get_model(
        param_img_size=param_img_size,
        param_frame=param_frame,
        device=device,
        model_name=model_name,
        output_dim=output_dim,
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device.index],
        output_device=device.index,
        broadcast_buffers=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    criterion = nt_xent_loss
    epoch_losses = []

    debug_shape_printed = False
    train_start = time.time()

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(
            data_loader,
            unit="batch",
            disable=(rank != 0),
            desc=f"[{framework_name}/{model_name}] Epoch {epoch + 1}/{epochs}",
        )

        for _, data in enumerate(progress_bar):
            inputs, labels = data
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if rank == 0 and not debug_shape_printed:
                print(f"[Debug] First batch input shape: {tuple(inputs.shape)}")
                print(f"[Debug] First batch label shape: {tuple(labels.shape)}")
                print(f"[Debug] Model input convention: [B, T, C, H, W]")
                if model_name != "ViViT":
                    print(f"[Debug] {model_name} internally permutes to [B, C, T, H, W]")
                debug_shape_printed = True

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(labels, outputs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
        epoch_loss_avg = epoch_loss_tensor.item() / world_size / len(data_loader)

        if rank == 0:
            epoch_losses.append(epoch_loss_avg)
            print(f"[Epoch {epoch + 1}/{epochs}] Loss: {epoch_loss_avg:.6f}")

    if rank == 0:
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(1, epochs + 1), epoch_losses, marker="o")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path_dict["loss_png"])
        plt.close()

        torch.save(model.module.state_dict(), path_dict["ckpt"])

        elapsed = time.time() - train_start
        print(f"[Train Complete] framework={framework_name} | model={model_name} | elapsed={elapsed:.2f}s")
        print(f"[Saved] {path_dict['loss_png']}")
        print(f"[Saved] {path_dict['ckpt']}")

    dist.barrier()
    return path_dict


def sil_score(represent_vectors, sil_save_path):
    sil_score_list = []

    for i in tqdm(range(2, 21), desc="[Eval] Silhouette search"):
        km = KMeans(n_clusters=i, max_iter=1000, init="k-means++", n_init=10)

        normed_vectors = preprocessing.normalize(represent_vectors)
        km.fit(normed_vectors)
        pred = km.predict(normed_vectors)

        if len(set(pred)) == 1:
            sil_score_list.append([i, 0])
            continue

        score = silhouette_score(represent_vectors, labels=pred, metric="cosine")
        sil_score_list.append([i, score])

    result = pd.DataFrame(data=sil_score_list, columns=["n_clusters", "silhouette_score"])
    pivot_km = pd.pivot_table(result, index="n_clusters", values="silhouette_score")

    plt.figure(figsize=(6, 8))
    sns.heatmap(pivot_km, annot=True, linewidths=0.5, fmt=".3f", cmap=sns.cm._rocket_lut)
    plt.tight_layout()
    plt.savefig(sil_save_path)
    plt.close()

    best_cluster_n = sorted(sil_score_list, key=lambda x: x[1], reverse=True)[0][0]
    print(f"[Eval] Best cluster count by silhouette score: {best_cluster_n}")
    print(f"[Saved] {sil_save_path}")
    return best_cluster_n


def eval_model(
    param_frame,
    param_img_size,
    dataset_dir,
    device,
    model_name,
    framework_name,
    save_dir,
    output_dim,
    original_video_dir=None,
):
    path_dict = get_paths(save_dir, framework_name, model_name)

    model = get_model(
        param_img_size=param_img_size,
        param_frame=param_frame,
        device=device,
        model_name=model_name,
        output_dim=output_dim,
    )

    state_dict = torch.load(path_dict["ckpt"], map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    represented_result = []
    dataset = VideoDataset(data_dir=dataset_dir, frame=param_frame, img_size=param_img_size)

    print("=" * 80)
    print(f"[Start Eval] framework={framework_name} | model={model_name}")
    print(f"[Eval Dataset dir] {dataset_dir}")
    print("=" * 80)

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(dataset, desc=f"[{framework_name}/{model_name}] Eval")):
            sample = sample / 255.0
            sample = torch.from_numpy(np.expand_dims(sample, axis=0)).to(device)  # [1, T, H, W, C]
            sample = sample.permute(0, 1, 4, 2, 3).contiguous()  # [1, T, C, H, W]

            represented_vector = model(sample)
            represented_result.append(represented_vector.detach().cpu().numpy())

            if idx == 0:
                print(f"[Debug Eval] First eval sample input shape: {tuple(sample.shape)}")
                print(f"[Debug Eval] First eval output shape: {tuple(represented_vector.shape)}")

    represented_result = np.squeeze(np.array(represented_result), axis=1)
    best_cluster_n = sil_score(represented_result, path_dict["sil_png"])

    km = KMeans(n_clusters=best_cluster_n, max_iter=1000, init="k-means++", n_init=10)
    normed_result = preprocessing.normalize(represented_result)
    km.fit(normed_result)
    pred = km.predict(normed_result)

    df = pd.DataFrame(represented_result)
    df["cluster"] = pred
    df["data"] = dataset.data_list
    df.to_csv(path_dict["cluster_csv"], index=False)

    centroids = km.cluster_centers_
    centroid_df = pd.DataFrame(centroids)
    centroid_df.to_csv(path_dict["centroid_csv"], index=False)

    print(f"[Saved] {path_dict['cluster_csv']}")
    print(f"[Saved] {path_dict['centroid_csv']}")

    umap_visualization(
        csv_path=path_dict["cluster_csv"],
        output_path=path_dict["umap_png"],
    )

    if original_video_dir is not None:
        extract_video(
            video_dir=original_video_dir,
            saved_result_csv=path_dict["cluster_csv"],
            extracted_video_dir=path_dict["extracted_video_dir"],
            top_k=5,
        )

    print(f"[Eval Complete] framework={framework_name} | model={model_name}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--frame", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--dataset_dir", type=str, default="./datasets/video_obj_resized")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument(
        "--model_name",
        type=str,
        default="ViViT",
        choices=["ViViT", "C3D", "X3D"],
    )
    parser.add_argument("--framework_name", type=str, default="simclr")
    parser.add_argument("--save_dir", type=str, default="./experiments/simclr_ViViT")
    parser.add_argument("--output_dim", type=int, default=4)

    parser.add_argument(
        "--original_video_dir",
        type=str,
        default=None,
        help="Optional. If provided, representative videos will be copied after clustering.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    rank, local_rank, world_size, device = setup_ddp()

    try:
        train(
            param_frame=args.frame,
            param_img_size=args.img_size,
            dataset_dir=args.dataset_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            device=device,
            model_name=args.model_name,
            framework_name=args.framework_name,
            save_dir=args.save_dir,
            num_workers=args.num_workers,
            output_dim=args.output_dim,
            run_args=args,
        )

        if rank == 0:
            eval_model(
                param_frame=args.frame,
                param_img_size=args.img_size,
                dataset_dir=args.dataset_dir,
                device=device,
                model_name=args.model_name,
                framework_name=args.framework_name,
                save_dir=args.save_dir,
                output_dim=args.output_dim,
                original_video_dir=args.original_video_dir,
            )

    finally:
        cleanup_ddp()