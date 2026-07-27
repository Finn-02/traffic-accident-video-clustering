# Traffic Accident Video Analysis using MoCo v3 and Object Detection

Self-supervised traffic accident video analysis using **MoCo v3**, **object detection-based filtering**, and **cluster-based scenario analysis** on the **AIHub traffic accident video dataset**.

---

## Overview

This repository provides an end-to-end experimental pipeline for analyzing traffic accident videos with self-supervised learning.  
Starting from raw AIHub traffic accident videos, the pipeline constructs a standardized dataset, applies object detection-based filtering, learns video representations with contrastive learning, clusters similar accident scenarios, and supports qualitative and metadata-driven cluster analysis.

The overall pipeline is:

**AIHub traffic accident dataset**  
→ **dataset construction and video standardization**  
→ **object detection-based filtering**  
→ **self-supervised contrastive learning**  
→ **K-Means clustering with silhouette score search**  
→ **UMAP visualization**  
→ **optional cluster-level semantic/statistical analysis**

---

## Key Features

- Preprocessing pipeline for the **AIHub traffic accident video dataset**
- Construction of a standardized **10-second black box video subset**
- **Object detection-based masking** using DETR
- Self-supervised training with **MoCo v3 / SimCLR-style contrastive learning**
- Support for multiple video encoders such as **ViViT, C3D, X3D, R3D18, MC3_18, and R2Plus1D**
- Automatic **K-Means clustering** with **silhouette score-based cluster number selection**
- **UMAP**-based embedding visualization
- Extraction of **representative videos** and **failure-case videos**
- Optional **coarse-grained label analysis** for cluster interpretation

---
## Repository Structure

```bash
VideoContrastive/
├── data/
│   ├── Datasets/
│   │   ├── createDataset.py
│   │   └── objectDetection.py
│   └── analysis/
│       ├── build_analysis_table_coarse.py
│       ├── cluster_signature_analysis.py
│       ├── plot_first_order_coarse.py
│       └── plot_second_order_coarse.py
└── code/
    ├── moco3/
    │   ├── dataloader.py
    │   ├── extract_video.py
    │   ├── losses.py
    │   ├── models.py
    │   ├── train.py
    │   └── umap_visualization.py
    └── simclr/
        ├── dataloader.py
        ├── extract_video.py
        ├── losses.py
        ├── models.py
        ├── train.py
        └── umap_visualization.py
```

### Main Files

* **createDataset.py**
  Builds the experimental dataset from the original AIHub download. It unzips files, filters the target subset, resizes videos, cuts them to fixed duration, and validates the results.

* **objectDetection.py**
  Applies DETR-based object filtering to accident videos and creates object-focused videos for training.

* **dataloader.py**
  Defines the dataset, frame sampling/padding logic, resizing, and tensor-based augmentation.

* **models.py**
  Contains the video encoders used in the experiments.

* **losses.py**
  Defines the contrastive loss used during training.

* **train.py**
  Multi-GPU training, clustering, and visualization pipeline. This is the main script for reproducing the core experiments.

* **umap_visualization.py**
  Generates 2D UMAP projections from clustering results.

* **extract_video.py**
  Extracts representative top videos and low-similarity bottom videos per cluster.

* **build_analysis_table_coarse.py**
  Builds a coarse-grained analysis table from a merged fine-grained analysis table.

* **plot_first_order_coarse.py**
  Runs first-order coarse label analysis.

* **plot_second_order_coarse.py**
  Runs second-order coarse label-pair analysis.

* **cluster_signature_analysis.py**
  Computes cluster signatures using standardized residuals and pair enrichment analysis.

---

## Dataset

This project uses the **AIHub traffic accident video dataset**.

The experimental setting focuses on:

* **black box videos**
* **1st-person point of view**
* **vehicle-centered accident clips**
* **10-second standardized videos**
* frame-standardized inputs for self-supervised learning
* object-focused filtering before representation learning

### Expected AIHub Folder Structure

Before running the preprocessing script, the downloaded AIHub dataset should preserve the original folder structure:

```bash
<DATASET_ROOT>/
└── 01.데이터/
    ├── 1.Training/
    │   ├── 원천데이터_231108_add/
    │   └── 라벨링데이터_231108_add/
    └── 2.Validation/
        ├── 원천데이터_231108_add/
        └── 라벨링데이터_231108_add/
```

---

## Environment

### Recommended

* Python 3.10+
* PyTorch
* torchvision
* transformers
* scikit-learn
* pandas
* numpy
* opencv-python
* matplotlib
* seaborn
* umap-learn
* einops
* tqdm

### Example Installation

```bash
pip install torch torchvision transformers scikit-learn pandas numpy opencv-python matplotlib seaborn umap-learn einops tqdm
```

---

## Experimental Pipeline

## 1. Dataset Construction

First, edit the hard-coded paths at the top of `createDataset.py`.

```python
DATASET_ORIGINAL_FOLRDER_PATH = "/path/to/AIHub/01.데이터"
SAVE_VIDEO_FOLRDER = "/path/to/video_original"
SAVE_RESIZED_VIDEO_FOLRDER = "/path/to/video_resized"
SAVE_LABEL_FOLRDER = "/path/to/label"
```

Then run:

```bash
python createDataset.py
```

### What this script does

1. checks whether the original AIHub folders exist
2. unzips all video and label archives
3. filters videos by metadata
4. keeps only videos satisfying:

   * `filming_way == 'bb'`
   * `video_point_of_view == 1`
5. resizes videos to **1920 × 1080**
6. keeps only the first **10 seconds**
7. validates frame counts and removes invalid outputs

### Important Notes

* `createDataset.py` expects the save folders to **not already exist**.
* If `SAVE_VIDEO_FOLRDER`, `SAVE_RESIZED_VIDEO_FOLRDER`, or `SAVE_LABEL_FOLRDER` already exist, the script stops with an error.
* In the original experimental setting, the script comments indicate the following final counts:

```text
video_original total: 14212
label total: 14212
videos shorter than 10 sec: 3
videos removed due to frame mismatch: 14
final video_resized total: 14195
```

### Expected Output

```bash
/path/to/datasets/
├── video_original/
├── video_resized/
└── label/
```

---

## 2. Object Detection-Based Filtering

Next, edit the hard-coded paths at the top of `objectDetection.py`.

```python
SAVE_RESIZED_VIDEO_FOLRDER = "/path/to/video_resized"
SAVE_OBJFILTER_VIDEO_FOLDER = "/path/to/video_obj"
SAVE_OBJFILTER_RESIZED_VIDEO_FOLDER = "/path/to/video_obj_resized"
```

Then run:

```bash
python objectDetection.py
```

### What this script does

* loads videos from `video_resized`
* samples frames from each video
* applies pretrained **DETR** object detection
* keeps only detected traffic-relevant objects
* masks other regions
* saves object-filtered videos
* resizes the filtered outputs to **224 × 224**

### Object Categories Used

The current implementation keeps only these DETR labels:

* `car`
* `truck`
* `bus`
* `traffic light`

### Important Notes

* The current implementation is written for **2 GPUs** and explicitly launches:

  * `cuda:0`
  * `cuda:1`
* If you only have **1 GPU**, you need to modify `objectDetection.py` before running it.
* The script deletes and recreates the output folders `video_obj` and `video_obj_resized`.

### Expected Output

```bash
/path/to/datasets/
├── video_obj/
└── video_obj_resized/
```

---

## 3. Contrastive Training, Clustering, and Visualization

The main experiment script is `train.py`.

It performs:

* contrastive learning
* checkpoint saving
* representation extraction
* silhouette score search over cluster numbers
* K-Means clustering
* cluster result CSV saving
* centroid CSV saving
* UMAP visualization
* optional representative video extraction

### Recommended Multi-GPU Command

```bash
torchrun --nproc_per_node=2 train.py \
  --frame 16 \
  --img_size 224 \
  --dataset_dir /path/to/video_obj_resized \
  --epochs 100 \
  --batch_size 32 \
  --num_workers 8 \
  --model_name ViViT \
  --framework_name moco3 \
  --save_dir /path/to/experiments/moco3_ViViT_100 \
  --output_dim 4 \
  --original_video_dir /path/to/video_original
```

### Available `--model_name` options in `train.py`

* `ViViT`
* `R3D18`
* `MC3_18`
* `R2Plus1D`
* `C3D`
* `X3D`

### Main Arguments

* `--frame` : number of frames sampled from each video
* `--img_size` : frame resolution for model input
* `--dataset_dir` : path to the preprocessed object-filtered videos
* `--epochs` : number of training epochs
* `--batch_size` : global batch size
* `--num_workers` : number of dataloader workers
* `--model_name` : video backbone
* `--framework_name` : experiment name tag, e.g. `moco3` or `simclr`
* `--save_dir` : output directory for experiment results
* `--output_dim` : output feature dimension of the representation head
* `--original_video_dir` : optional path to the original videos for representative video extraction after clustering

---

## 4. Main Outputs

After training and evaluation, the experiment directory contains outputs such as:

```bash
<save_dir>/
├── moco3_ViViT_Training_Loss.png
├── moco3_ViViT_ckpt.weights.pth
├── moco3_ViViT_cluster_result.csv
├── moco3_ViViT_cluster_centers.csv
├── moco3_ViViT_silhouette_score.png
├── moco3_ViViT_umap_projection_2d.png
├── moco3_ViViT_run_config.json
└── moco3_ViViT_extracted_video/
```

### Output Description

* `*_Training_Loss.png`
  training loss curve

* `*_ckpt.weights.pth`
  trained model checkpoint

* `*_cluster_result.csv`
  learned video representations with cluster assignments and file names

* `*_cluster_centers.csv`
  K-Means centroids

* `*_silhouette_score.png`
  silhouette score heatmap used for cluster number selection

* `*_umap_projection_2d.png`
  2D UMAP projection of clustered embeddings

* `*_run_config.json`
  saved experiment configuration

* `*_extracted_video/`
  representative top videos and low-similarity bottom videos for qualitative analysis

---

## 5. Extract Representative Videos Separately

If you already have a cluster result CSV and want to extract representative or failure-case videos separately, use:

```bash
python extract_video.py \
  --video_dir /path/to/video_original \
  --saved_result_csv /path/to/experiments/moco3_ViViT_100/moco3_ViViT_cluster_result.csv \
  --extracted_video_dir /path/to/experiments/moco3_ViViT_100/moco3_ViViT_extracted_video \
  --top_k 5 \
  --bottom_k 20
```

### What this script does

For each cluster:

* computes a centroid from normalized feature vectors
* ranks videos by cosine similarity to the centroid
* copies:

  * top-k most representative videos
  * bottom-k least representative videos

This is useful for:

* qualitative cluster inspection
* representative accident pattern visualization
* failure-case analysis

---

## 6. Optional Cluster-Level Semantic Analysis

This repository also includes post-processing scripts for **coarse-grained cluster interpretation**.

These scripts are **not part of the basic training pipeline**. They are used **after** you have already prepared a merged analysis table that combines:

* cluster assignments
* accident metadata
* coarse/fine label information

In other words, these analysis scripts assume that a file such as `merged_analysis_table.csv` already exists.

### 6-1. Build Coarse Analysis Table

```bash
python build_analysis_table_coarse.py \
  --experiment moco3 \
  --model ViViT \
  --epoch 100 \
  --input-csv /path/to/merged_analysis_table.csv \
  --output-dir /path/to/analysis_output_coarse/moco3_ViViT_100
```

### 6-2. First-Order Coarse Analysis

```bash
python plot_first_order_coarse.py \
  --experiment moco3 \
  --model ViViT \
  --epoch 100 \
  --input-csv /path/to/analysis_output_coarse/moco3_ViViT_100/merged_analysis_table_coarse.csv \
  --output-dir /path/to/analysis_output_coarse/moco3_ViViT_100/first_order_coarse
```

### 6-3. Second-Order Coarse Analysis

```bash
python plot_second_order_coarse.py \
  --experiment moco3 \
  --model ViViT \
  --epoch 100 \
  --input-csv /path/to/analysis_output_coarse/moco3_ViViT_100/merged_analysis_table_coarse.csv \
  --output-dir /path/to/analysis_output_coarse/moco3_ViViT_100/second_order_coarse \
  --top-k 20
```

### 6-4. Cluster Signature Analysis

```bash
python cluster_signature_analysis.py \
  --experiment moco3 \
  --model ViViT \
  --epoch 100 \
  --input-csv /path/to/analysis_output_coarse/moco3_ViViT_100/merged_analysis_table_coarse.csv \
  --output-dir /path/to/analysis_output_coarse/moco3_ViViT_100/cluster_signature_analysis \
  --top-k 5
```

These scripts support cluster interpretation using:

* coarse accident place categories
* coarse accident place feature categories
* negligence rate combinations
* vehicle A/B progress information groupings
* first-order frequency patterns
* second-order pair patterns
* standardized residual-based signature analysis

---

## Recommended Reproduction Order

For full reproduction, the recommended order is:

```text
1. Download the AIHub traffic accident video dataset
2. Edit paths in createDataset.py
3. Run python createDataset.py
4. Edit paths in objectDetection.py
5. Run python objectDetection.py
6. Run train.py
7. Inspect silhouette score, cluster CSV, and UMAP projection
8. Optionally extract representative videos
9. If metadata merge is available, run the coarse analysis scripts
```

---

## Quick Start

If you only want the core experiment pipeline, use this order:

```bash
python createDataset.py
python objectDetection.py
torchrun --nproc_per_node=2 train.py \
  --frame 16 \
  --img_size 224 \
  --dataset_dir /path/to/video_obj_resized \
  --epochs 100 \
  --batch_size 32 \
  --num_workers 8 \
  --model_name ViViT \
  --framework_name moco3 \
  --save_dir /path/to/experiments/moco3_ViViT_100 \
  --output_dim 4 \
  --original_video_dir /path/to/video_original
```

---

## Important Notes

* `createDataset.py` and `objectDetection.py` rely on **hard-coded paths**, so you must edit them before running.

* `objectDetection.py` is currently written for **2 GPUs**.

* `train.py` is the recommended script for the main experiments.

* The basic public pipeline of this repository is:

  **dataset construction → object filtering → contrastive training → clustering → visualization**

* The analysis scripts are optional and assume an already prepared merged metadata table.

* The current augmentation pipeline is implemented directly in `dataloader.py`.

---

## Citation

If you use this repository in your research, please cite the corresponding paper:

**Traffic Accident Video Analysis using MoCo v3 and Object Detection**

```bibtex
@article{traffic_accident_moco3_object_detection,
  title={Traffic Accident Video Analysis using MoCo v3 and Object Detection},
  author={Anonymous},
  journal={},
  year={}
}
```

---

## Acknowledgement

This work is based on the **AIHub traffic accident video dataset** and studies self-supervised video representation learning, clustering, and cluster interpretation for traffic accident analysis.
