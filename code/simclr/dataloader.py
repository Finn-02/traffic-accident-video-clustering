import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _sample_or_pad_frames(frames: np.ndarray, target_frame: int) -> np.ndarray:
    """
    입력 frames: [T, H, W, C]
    출력: 항상 [target_frame, H, W, C]
    """
    num_frames = len(frames)

    if num_frames == 0:
        raise ValueError("Video has 0 frames.")

    if num_frames == target_frame:
        return frames

    if num_frames > target_frame:
        indices = np.linspace(0, num_frames - 1, target_frame).astype(np.int32)
        return frames[indices]

    pad_count = target_frame - num_frames
    pad_frames = np.repeat(frames[-1][None, ...], pad_count, axis=0)
    return np.concatenate([frames, pad_frames], axis=0)


def _random_resized_crop_tensor(video_tchw: torch.Tensor, out_h: int, out_w: int, scale=(0.75, 0.9)) -> torch.Tensor:
    """
    video_tchw: [T, C, H, W]
    """
    t, c, h, w = video_tchw.shape
    area = h * w

    for _ in range(10):
        target_area = np.random.uniform(scale[0], scale[1]) * area
        aspect_ratio = 1.0

        crop_h = int(round(np.sqrt(target_area / aspect_ratio)))
        crop_w = int(round(np.sqrt(target_area * aspect_ratio)))

        if 0 < crop_h <= h and 0 < crop_w <= w:
            top = np.random.randint(0, h - crop_h + 1)
            left = np.random.randint(0, w - crop_w + 1)

            cropped = video_tchw[:, :, top:top + crop_h, left:left + crop_w]
            resized = F.interpolate(
                cropped,
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            )
            return resized

    return F.interpolate(video_tchw, size=(out_h, out_w), mode="bilinear", align_corners=False)


def augment_video(video: np.ndarray, img_size: int) -> torch.Tensor:
    """
    입력: [T, H, W, C], np.float32, RGB, [0,255]
    출력: [T, C, H, W], torch.float32, [0,1]
    """
    num_images, height, width, channels = video.shape

    mean = 0.0
    var = 0.01
    sigma = var ** 0.5

    brightness_factor = np.random.uniform(0.7, 1.3)
    contrast_factor = np.random.uniform(0.8, 1.2)
    gaussian = np.random.normal(mean, sigma, video.shape).astype(np.float32)

    augmented = video.copy()

    for i in range(num_images):
        image = augmented[i]

        image = np.clip(image + gaussian[i] * 255.0, 0, 255)
        image = np.clip(image * brightness_factor, 0, 255)
        image = (image - 127.5) * contrast_factor + 127.5
        image = np.clip(image, 0, 255)

        if np.random.rand() < 0.5:
            image = np.ascontiguousarray(np.fliplr(image))

        augmented[i] = image

    augmented = augmented / 255.0
    video_pt = torch.from_numpy(augmented).permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]

    video_pt = _random_resized_crop_tensor(video_pt, img_size, img_size, scale=(0.75, 0.9))

    if np.random.rand() < 0.2:
        gray = (
            0.2989 * video_pt[:, 0:1] +
            0.5870 * video_pt[:, 1:2] +
            0.1140 * video_pt[:, 2:3]
        )
        video_pt = gray.repeat(1, 3, 1, 1)

    if np.random.rand() < 0.2:
        video_pt = F.avg_pool2d(video_pt, kernel_size=3, stride=1, padding=1)

    video_pt = torch.clamp(video_pt, 0.0, 1.0)
    return video_pt.float()


def build_collate_fn(img_size: int):
    def collate_fn(samples):
        """
        samples: list of [T, H, W, C]
        반환:
          augmented_x: [2B, T, C, H, W]
          labels:      [2B]
        """
        aug_1 = [augment_video(data.copy(), img_size=img_size) for data in samples]
        aug_2 = [augment_video(data.copy(), img_size=img_size) for data in samples]

        aug_result = aug_1 + aug_2
        augmented_x = torch.stack(aug_result).float()

        batch_size = len(samples)
        labels = np.concatenate(
            [np.arange(batch_size, batch_size * 2), np.arange(0, batch_size)],
            axis=0,
        )
        labels = torch.from_numpy(labels).long()

        return augmented_x, labels

    return collate_fn


class VideoDataset(Dataset):
    def __init__(self, data_dir: str, frame: int, img_size: int):
        self.data_dir = data_dir
        self.data_list = sorted(
            [
                f for f in os.listdir(data_dir)
                if os.path.isfile(os.path.join(data_dir, f))
            ]
        )
        self.frame = frame
        self.img_size = img_size

        if len(self.data_list) == 0:
            raise ValueError(f"No files found in data_dir: {data_dir}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]
        full_path = os.path.join(self.data_dir, file_name)

        frames = []
        cap = cv2.VideoCapture(full_path)

        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {full_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No readable frames in video: {full_path}")

        x = np.array(frames, dtype=np.float32)  # [T, H, W, C]
        x = _sample_or_pad_frames(x, self.frame)

        return x