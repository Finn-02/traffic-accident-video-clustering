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


class VideoDataset(Dataset):
    def __init__(self, data_dir: str, frame: int = 16, img_size: int = 224):
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
        x = x / 255.0

        # [T, H, W, C] -> [T, C, H, W]
        x = np.transpose(x, (0, 3, 1, 2)).copy()

        return torch.from_numpy(x).float()


class Augmentator:
    """
    pytorchvideo 의존 없이 동작하는 간단한 tensor augmentation.
    입력 x: [N, C, H, W], 값 범위 [0, 1]
    """
    def __init__(
        self,
        device,
        p_flip: float = 0.5,
        brightness: float = 0.2,
        contrast: float = 0.2,
        grayscale_p: float = 0.1,
        blur_p: float = 0.2,
        noise_std: float = 0.02,
    ):
        self.device = device
        self.p_flip = p_flip
        self.brightness = brightness
        self.contrast = contrast
        self.grayscale_p = grayscale_p
        self.blur_p = blur_p
        self.noise_std = noise_std

    def _horizontal_flip(self, x):
        flip_mask = (torch.rand(x.size(0), device=x.device) < self.p_flip)
        if flip_mask.any():
            x[flip_mask] = torch.flip(x[flip_mask], dims=[3])
        return x

    def _color_jitter(self, x):
        n = x.size(0)

        brightness_factor = 1.0 + (torch.rand(n, 1, 1, 1, device=x.device) * 2 - 1) * self.brightness
        contrast_factor = 1.0 + (torch.rand(n, 1, 1, 1, device=x.device) * 2 - 1) * self.contrast

        x = x * brightness_factor

        mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mean) * contrast_factor + mean
        return x

    def _grayscale(self, x):
        gray_mask = (torch.rand(x.size(0), device=x.device) < self.grayscale_p)
        if gray_mask.any():
            gray = (
                0.2989 * x[gray_mask, 0:1] +
                0.5870 * x[gray_mask, 1:2] +
                0.1140 * x[gray_mask, 2:3]
            )
            x[gray_mask] = gray.repeat(1, 3, 1, 1)
        return x

    def _blur(self, x):
        blur_mask = (torch.rand(x.size(0), device=x.device) < self.blur_p)
        if blur_mask.any():
            x[blur_mask] = F.avg_pool2d(x[blur_mask], kernel_size=3, stride=1, padding=1)
        return x

    def _noise(self, x):
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        return x

    def aug(self, x):
        """
        x: [N, C, H, W]
        """
        if x.ndim != 4:
            raise ValueError(f"Expected [N, C, H, W], got {tuple(x.shape)}")

        x = x.to(self.device, non_blocking=True)
        x = x.clone()

        x = self._horizontal_flip(x)
        x = self._color_jitter(x)
        x = self._grayscale(x)
        x = self._blur(x)
        x = self._noise(x)

        x = torch.clamp(x, 0.0, 1.0)
        return x