import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class NexarDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        frame_idx: int | None = None,
        return_label: bool = True,
        transform=None,
    ) -> None:
        self.df = df
        self.frame_idx = frame_idx
        self.return_label = return_label
        self.transform = transform

        self.features_path = df["features_path"].values
        self.n_frames = df["n_frames"].values
        if self.return_label:
            self.labels = df["target"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        data = {"idx": idx}

        # Sample frame
        frame_idx = (
            self.frame_idx
            if self.frame_idx is not None
            else np.random.randint(0, self.n_frames[idx] - 1)
        )
        data["frame_idx"] = frame_idx

        # Load features
        folder = self.features_path[idx]
        frame = torch.load(folder + f"/frames/{str(frame_idx).zfill(2)}.pt")
        flow = torch.load(folder + f"/flows/{str(frame_idx).zfill(2)}.pt")
        try:
            mask = torch.load(folder + f"/masks/{str(frame_idx).zfill(2)}.pt")
        except FileNotFoundError:
            mask = torch.zeros((1, *flow.shape[1:]))

        # Multiply by mask
        frame = frame * (mask > 0).float()
        flow = flow * (mask > 0).float()
        mask_flow = torch.cat([flow, mask], dim=0)

        # Apply transformations
        if self.transform:
            frame = self.apply_transform(frame)
            mask_flow = self.apply_transform(mask_flow)
        data["frame"] = frame.to(torch.float32)
        data["mask_flow"] = mask_flow.to(torch.float32)

        if self.return_label:
            data["label"] = self.labels[idx]

        return data

    def apply_transform(self, image: torch.Tensor) -> torch.Tensor:
        if image.dtype != torch.float32:
            image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        return self.transform(image)


class NexarDataModule(LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        batch_size: int = 32,
        val_size: float | None = None,
        transform=None,
        test_transform=None,
    ) -> None:
        super().__init__()
        self.train_df = train_df
        self.batch_size = batch_size
        self.val_size = val_size
        self.transform = transform
        self.test_transform = test_transform

    def setup(self, stage=None) -> None:
        if self.val_size is not None:
            train_df, val_df = train_test_split(
                self.train_df, test_size=self.val_size, stratify=self.train_df["target"]
            )
            self.train_dataset = NexarDataset(train_df, transform=self.transform)
            self.val_dataset = NexarDataset(val_df, transform=self.test_transform)

        self.train_dataset = NexarDataset(self.train_df, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader | None:
        if not hasattr(self, "val_dataset"):
            return None
        return DataLoader(self.val_dataset, batch_size=self.batch_size * 2)


def pad_to_square(image: torch.Tensor):
    """Pad the image to a square shape by adding zeros to all sides."""
    _, h, w = image.shape
    max_dim = max(w, h)
    pad_w = (max_dim - w) // 2
    pad_h = (max_dim - h) // 2

    # Padding format for torch.nn.functional.pad is (left, right, top, bottom)
    padding = (pad_w, max_dim - w - pad_w, pad_h, max_dim - h - pad_h)

    # Pad expects input as (N, C, H, W) or (C, H, W)
    return nn.functional.pad(image, padding, mode="constant", value=0)
