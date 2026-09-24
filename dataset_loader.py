#!/usr/bin/env python3
"""
dataset_loader.py: Module nạp dữ liệu PyTorch cực nhanh từ file HDF5 (.h5) hoặc ImageFolder.
Hỗ trợ DataLoader đa tiến trình (multi-worker) an toàn, tối ưu bộ nhớ khi load 1M - 2M mẫu.
Kèm sẵn một kiến trúc Convolutional Neural Network (CNN) chuẩn chỉnh để train ngay lập tức.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import h5py
from PIL import Image


# ==========================================
# 1. DATASET CHO ĐỊNH DẠNG HDF5 (.h5)
# ==========================================
class CharacterH5Dataset(Dataset):
    """
    Dataset nạp trực tiếp từ file HDF5 (.h5).
    Sử dụng lazy opening để an toàn 100% khi chạy DataLoader với num_workers > 0.
    """
    def __init__(self, h5_path: str, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        self.h5_file = None
        self.images = None
        self.labels = None

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Không tìm thấy file HDF5: {h5_path}")

        # Đọc độ dài ban đầu rồi đóng file ngay để không bị xung đột tiến trình
        with h5py.File(h5_path, "r") as f:
            self.length = len(f["labels"])

    def _init_h5(self):
        """Mở file HDF5 riêng biệt cho từng worker process."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
            self.images = self.h5_file["images"]
            self.labels = self.h5_file["labels"]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        self._init_h5()

        img_np = self.images[idx]  # uint8, shape: (H, W, C)
        label = int(self.labels[idx])

        # Chuyển sang PIL hoặc PyTorch Tensor
        if img_np.shape[-1] == 1:
            img = Image.fromarray(img_np[:, :, 0], mode="L")
        else:
            img = Image.fromarray(img_np, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)
        else:
            # Transform mặc định nếu chưa truyền: ToTensor chuẩn hoá về [0.0, 1.0]
            img = T.functional.to_tensor(img)

        return img, label

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None


# ==========================================
# 2. DATASET CHO ĐỊNH DẠNG IMAGE FOLDER / CSV
# ==========================================
class CharacterFolderDataset(Dataset):
    """Dataset nạp từ file manifest.csv hoặc thư mục ảnh."""
    def __init__(self, manifest_csv: str, base_dir: str, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        self.samples = []

        with open(manifest_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[1:]:  # Bỏ qua header
                parts = line.strip().split(",")
                if len(parts) == 2:
                    self.samples.append((parts[0], int(parts[1])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_path, label = self.samples[idx]
        full_path = os.path.join(self.base_dir, rel_path)
        img = Image.open(full_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = T.functional.to_tensor(img)

        return img, label

# 3. HÀM TẠO DATALOADER NHANH
# ==========================================
def get_dataloaders(data_dir: str, batch_size: int = 128, num_workers: int = 4,
                    format_type: str = "h5"):
    """
    Tạo train_loader và val_loader tối ưu.
    """
    # Transform chuẩn hoá
    train_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if format_type == "h5":
        train_h5 = os.path.join(data_dir, "dataset_train.h5")
        val_h5 = os.path.join(data_dir, "dataset_val.h5")
        train_ds = CharacterH5Dataset(train_h5, transform=train_transform)
        val_ds = CharacterH5Dataset(val_h5, transform=val_transform)
    else:
        train_csv = os.path.join(data_dir, "train_manifest.csv")
        val_csv = os.path.join(data_dir, "val_manifest.csv")
        train_ds = CharacterFolderDataset(train_csv, data_dir, transform=train_transform)
        val_ds = CharacterFolderDataset(val_csv, data_dir, transform=val_transform)

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0)
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=(num_workers > 0)
    )
    return train_loader, val_loader


# ==========================================
# 4. MODEL CNN MẪU (CHARACTER CLASSIFIER)
# ==========================================
class CharacterCNN(nn.Module):
    """
    Mạng Convolutional Neural Network hiện đại, gọn nhẹ:
    - 4 Conv Blocks kèm BatchNorm + ReLU + MaxPool + Dropout
    - Phù hợp phân loại 62 hoặc 36 class kí tự.
    """
    def __init__(self, num_classes: int = 62, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1),

            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),

            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),

            # Block 4: 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        out = self.classifier(feat)
        return out


# ==========================================
# 5. TEST & DEMO NHANH KHI CHẠY TRỰC TIẾP
# ==========================================
if __name__ == "__main__":
    print("[*] Kiểm tra kiến trúc model CharacterCNN...")
    model = CharacterCNN(num_classes=62, in_channels=3)
    dummy_input = torch.randn(4, 3, 64, 64)
    output = model(dummy_input)
    print(f"Model khởi tạo thành công! Input: {dummy_input.shape} -> Output: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    - Tổng tham số có thể train (trainable parameters): {total_params:,}")
