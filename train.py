#!/usr/bin/env python3
"""
train.py: File huấn luyện mạng nơ-ron (Convolutional Neural Network - CNN)
nhận diện kí tự đơn lẻ (62 class: 0-9, A-Z, a-z) từ dataset HDF5 2M mẫu.

Được chú thích cực kì chi tiết từ A-Z để m dễ dàng chỉnh sửa, thay đổi kiến trúc
hoặc tinh chỉnh hyperparameters (learning rate, batch size, số epoch,...).
"""

import os
import sys
import json
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import h5py
from tqdm import tqdm


# ==============================================================================
# 1. KHU VỰC CẤU HÌNH (HYPERPARAMETERS) - M TINH CHỈNH Ở ĐÂY NHA!
# ==============================================================================
# Đường dẫn dữ liệu
DATA_DIR = "./dataset"
TRAIN_H5_PATH = os.path.join(DATA_DIR, "dataset_train.h5")
VAL_H5_PATH = os.path.join(DATA_DIR, "dataset_val.h5")
CLASSES_JSON_PATH = os.path.join(DATA_DIR, "classes.json")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "best_model.pth")

# Tham số huấn luyện (m có thể tăng giảm tuỳ ý)
BATCH_SIZE = 256          # Số ảnh nạp vào model mỗi lượt (256 hoặc 512 chạy GPU Mac M4 rất mượt)
LEARNING_RATE = 1e-3      # Tốc độ học (0.001 là chuẩn, nếu thấy loss dao động mạnh thì giảm về 3e-4)
NUM_EPOCHS = 10           # Số lần lặp qua toàn bộ dataset (với 1.8M mẫu thì 5-10 epoch là acc > 95% rồi)
NUM_WORKERS = 4           # Số luồng CPU đọc data song song (Mac M4 để 4 hoặc 6 là vừa đẹp)
WEIGHT_DECAY = 1e-4       # Hệ số phạt L2 regularization để chống overfitting (học vẹt)


# ==============================================================================
# 2. CHỌN THIẾT BỊ TÍNH TOÁN (GPU APPLE SILICON / CUDA / CPU)
# ==============================================================================
def get_computing_device() -> torch.device:
    """Tự động kiểm tra và ưu tiên GPU Apple Silicon (chip M4 của m)."""
    if torch.backends.mps.is_available():
        print("Phát hiện GPU Apple Silicon! Sử dụng backend MPS (Metal).")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"Phát hiện GPU NVIDIA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("Không có GPU, chạy tạm bằng CPU nha.")
        return torch.device("cpu")


# ==============================================================================
# 3. CLASS NẠP DỮ LIỆU TỪ FILE HDF5 (.h5)
# ==============================================================================
class CharacterH5Dataset(Dataset):
    """
    Dataset tối ưu đọc file HDF5 khổng lồ.
    MẸO KỸ THUẬT: Không mở file h5 trong __init__, mà mở 'lười' (lazy) trong __getitem__.
    Lý do: Khi DataLoader chạy đa luồng (num_workers > 0), mỗi worker là 1 tiến trình riêng,
    nếu dùng chung file pointer sẽ bị crash hoặc đọc nhầm data!
    """
    def __init__(self, h5_path: str, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        self.h5_file = None
        self.images = None
        self.labels = None

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Kh tìm thấy file: {h5_path}. M kiểm tra lại đường dẫn nha!")

        # Mở đọc tạm tổng số mẫu & số kênh màu rồi đóng ngay
        with h5py.File(h5_path, "r") as f:
            self.total_samples = len(f["labels"])
            self.channels = f["images"].shape[-1]

    def _lazy_init(self):
        """Mỗi worker process sẽ mở 1 kết nối file h5 độc lập."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
            self.images = self.h5_file["images"]
            self.labels = self.h5_file["labels"]

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        self._lazy_init()

        # Đọc mảng numpy uint8 (H, W, C) từ file h5
        img_np = self.images[idx]
        label = int(self.labels[idx])

        # Chuyển sang PIL Image: hỗ trợ cả grayscale (1 kênh) lẫn RGB (3 kênh)
        if self.channels == 1:
            img_pil = Image.fromarray(img_np[:, :, 0], mode="L")
        else:
            img_pil = Image.fromarray(img_np, mode="RGB")

        # transform luôn được truyền vào từ main()
        img_tensor = self.transform(img_pil)

        return img_tensor, label


# ==============================================================================
# 4. ĐỊNH NGHĨA KIẾN TRÚC MẠNG NƠ-RON (CNN - CONVOLUTIONAL NEURAL NETWORK)
# ==============================================================================
class CharacterClassifierCNN(nn.Module):
    """
    Kiến trúc CNN 4 tầng tiêu chuẩn cho nhận diện ảnh.

    SO SÁNH VỚI KHOÁ HỌC CỦA THẦY ANDREW NG:
    - Trong bài tập tuần 2, thầy Andrew dùng mạng Dense (MLP): duỗi phẳng ảnh rồi nhân ma trận W*X + b.
      Nhưng với ảnh màu 64x64x3 (12,288 pixels), nếu dùng Dense(128) sẽ tốn 1.5 TRIỆU tham số ngay layer 1!
    - CNN giải quyết bằng cách: Dùng các bộ lọc nhỏ 3x3 (Kernel) trượt khắp bức ảnh để bắt nét chữ.
      Cùng 1 bộ lọc dùng chung cho cả bức ảnh (Parameter Sharing) nên cực kì nhẹ và chuẩn xác!

    Ý NGHĨA TỪNG LỚP:
    - Conv2d: Bộ lọc trượt trên ảnh trích xuất nét cong, góc nhọn, đường thẳng.
    - BatchNorm2d: Chuẩn hoá dữ liệu giữa các layer giúp model hội tụ nhanh hơn.
    - ReLU: Hàm kích hoạt phi tuyến tính f(x) = max(0, x) y hệt thầy Andrew dạy.
    - MaxPool2d: Giảm kích thước ảnh 1 nửa (64 -> 32 -> 16 -> 8 -> 4) để cô đọng đặc trưng.
    - Dropout: Tắt ngẫu nhiên nơ-ron để chống Overfitting (học vẹt).
    - Linear: Chính là lớp 'Dense' của thầy Andrew để chấm điểm cho 62 class.
    """
    def __init__(self, num_classes: int = 62, in_channels: int = 3):
        super().__init__()

        # --- KHỐI TRÍCH XUẤT ĐẶC TRƯNG (FEATURE EXTRACTOR) ---
        self.features = nn.Sequential(
            # Tầng 1: Input (3, 64, 64) -> Output (32, 32, 32)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Giảm từ 64x64 xuống 32x32
            nn.Dropout2d(0.1),                      # Tắt ngẫu nhiên 10% channel

            # Tầng 2: Input (32, 32, 32) -> Output (64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Giảm từ 32x32 xuống 16x16
            nn.Dropout2d(0.15),

            # Tầng 3: Input (64, 16, 16) -> Output (128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Giảm từ 16x16 xuống 8x8
            nn.Dropout2d(0.2),

            # Tầng 4: Input (128, 8, 8) -> Output (256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Giảm từ 8x8 xuống 4x4
            nn.Dropout2d(0.25),
        )

        # --- KHỐI PHÂN LOẠI (CLASSIFIER) ---
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),   # Thu nhỏ về tensor (256, 1, 1)
            nn.Flatten(),                   # Trải phẳng thành vector 256 phần tử
            nn.Linear(256, 128),            # Nén từ 256 -> 128 nơ-ron
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),                # Tắt 40% nơ-ron ở tầng fully-connected
            nn.Linear(128, num_classes)     # Output ra 62 giá trị điểm (logits) của 62 class
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Định nghĩa luồng dữ liệu đi qua model."""
        x = self.features(x)
        x = self.classifier(x)
        return x


# ==============================================================================
# 5. HÀM TRAIN 1 EPOCH
# ==============================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    model.train()  # Bật chế độ training (kích hoạt Dropout & BatchNorm cập nhật mean/var)
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Train]", unit="batch")
    for images, labels in pbar:
        # Đẩy dữ liệu vào GPU
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ======================================================================
        # 5 BƯỚC HUẤN LUYỆN KINH ĐIỂN (TƯƠNG ĐƯƠNG MODEL.FIT TRONG TENSORFLOW)
        # ======================================================================
        # Bước 1: Xoá sạch đạo hàm dJ/dw của batch trước đó
        optimizer.zero_grad()

        # Bước 2: Forward Propagation (Tính giá trị dự đoán f_w,b(x))
        outputs = model(images)

        # Bước 3: Tính hàm mất mát J(w,b) (Cross-Entropy Loss)
        loss = criterion(outputs, labels)

        # Bước 4: Backpropagation (Lan truyền ngược để tính đạo hàm dJ/dw và dJ/db)
        loss.backward()

        # Bước 5: Gradient Descent (Cập nhật trọng số: w = w - learning_rate * dJ/dw)
        optimizer.step()
        # ======================================================================

        # Gom thống kê để tính Loss & Accuracy
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        _, predicted_classes = torch.max(outputs, dim=1)
        correct_predictions += (predicted_classes == labels).sum().item()
        total_samples += batch_size

        # Cập nhật thanh tiến trình theo thời gian thực
        current_loss = running_loss / total_samples
        current_acc = (correct_predictions / total_samples) * 100
        pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"})

    return running_loss / total_samples, (correct_predictions / total_samples) * 100


# ==============================================================================
# 6. HÀM ĐÁNH GIÁ (VALIDATION) TRÊN TẬP VAL
# ==============================================================================
@torch.no_grad()  # Tắt tính gradient để tiết kiệm RAM và tăng tốc
def evaluate_model(model, dataloader, criterion, device, epoch, total_epochs):
    model.eval()  # Tắt Dropout, cố định BatchNorm
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Val]  ", unit="batch")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        _, predicted_classes = torch.max(outputs, dim=1)
        correct_predictions += (predicted_classes == labels).sum().item()
        total_samples += batch_size

        current_loss = running_loss / total_samples
        current_acc = (correct_predictions / total_samples) * 100
        pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"})

    return running_loss / total_samples, (correct_predictions / total_samples) * 100


# ==============================================================================
# 7. HÀM TEST DỰ ĐOÁN THỬ MỘT VÀI ẢNH BẰNG MODEL ĐÃ TRAIN
# ==============================================================================
def demo_inference(model, val_dataset, idx_to_char, device, num_tests=5):
    """Bốc ngẫu nhiên vài ảnh từ tập Val ra cho model đoán thử."""
    print("\n" + "=" * 55)
    print("KẾT QUẢ DỰ ĐOÁN THỬ NGHIỆM TRÊN MỘT SỐ ẢNH VAL:")
    print("=" * 55)
    model.eval()

    import random
    indices = [random.randint(0, len(val_dataset) - 1) for _ in range(num_tests)]

    for i, idx in enumerate(indices, 1):
        img_tensor, true_label = val_dataset[idx]
        input_tensor = img_tensor.unsqueeze(0).to(device)  # Thêm chiều batch: (1, 3, 64, 64)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_label].item() * 100

        true_char = idx_to_char[str(true_label)]
        pred_char = idx_to_char[str(pred_label)]
        status = "ĐÚNG" if true_char == pred_char else "SAI"

        print(f"[{i}] Thực tế: '{true_char}'  |  Model đoán: '{pred_char}' ({confidence:.1f}%)  --> {status}")
    print("=" * 55)


# ==============================================================================
# 8. HÀM CHÍNH (MAIN ENTRY POINT)
# ==============================================================================
def main():
    print("=" * 65)
    print("BẮT ĐẦU CHƯƠNG TRÌNH HUẤN LUYỆN MODEL NHẬN DIỆN KÍ TỰ")
    print("=" * 65)

    # 1. Đọc metadata danh sách class
    if not os.path.exists(CLASSES_JSON_PATH):
        print(f"[X] Lỗi: Không tìm thấy '{CLASSES_JSON_PATH}'!")
        sys.exit(1)

    with open(CLASSES_JSON_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    num_classes = meta["num_classes"]
    idx_to_char = meta["idx_to_char"]
    channels = meta.get("channels", 3)  # Mặc định 3 cho dataset cũ thiếu key 'channels'
    print(f"[*] Số lượng kí tự cần phân loại: {num_classes} classes")
    print(f"[*] Danh sách class: {''.join(meta['classes'])}")

    # 2. Chuẩn bị Device
    device = get_computing_device()

    # 3. Chuẩn bị Transform (Chuẩn hoá pixel ảnh về phân phối chuẩn)
    if channels == 1:
        data_transform = T.Compose([
            T.ToTensor(),  # Đổi giá trị từ [0, 255] sang [0.0, 1.0]
            T.Normalize(mean=[0.5], std=[0.5]),  # Chuẩn cho ảnh grayscale
        ])
    else:
        data_transform = T.Compose([
            T.ToTensor(),  # Đổi giá trị từ [0, 255] sang [0.0, 1.0]
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Chuẩn ImageNet
        ])

    # 4. Khởi tạo Dataset & DataLoader
    print("[*] Đang nạp dataset HDF5...")
    train_dataset = CharacterH5Dataset(TRAIN_H5_PATH, transform=data_transform)
    val_dataset = CharacterH5Dataset(VAL_H5_PATH, transform=data_transform)
    print(f"Đã kết nối: Train ({len(train_dataset):,} mẫu) | Val ({len(val_dataset):,} mẫu)")

    # DataLoader tự động gom batch và shuffle
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,                # Trộn ngẫu nhiên data mỗi epoch
        num_workers=NUM_WORKERS,
        persistent_workers=(NUM_WORKERS > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=(NUM_WORKERS > 0)
    )

    # 5. Khởi tạo Mô hình, Hàm mất mát và Thuật toán tối ưu
    model = CharacterClassifierCNN(num_classes=num_classes, in_channels=channels).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Khởi tạo model thành công! Tổng số tham số (weights): {total_params:,}")

    # Hàm mất mát CrossEntropy cho phân loại nhiều class
    # (TƯƠNG ĐƯƠNG SparseCategoricalCrossentropy(from_logits=True) của thầy Andrew Ng)
    criterion = nn.CrossEntropyLoss()

    # Thuật toán AdamW (TƯƠNG ĐƯƠNG tf.keras.optimizers.Adam trong khoá học)
    # WEIGHT_DECAY chính là hệ số phạt L2 Regularization (lambda) chống Overfitting
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Lịch giảm dần learning rate (Cosine Annealing) để cuối quá trình model học tinh tế hơn
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 6. VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP)
    best_val_accuracy = 0.0
    start_time = time.time()

    print("\n" + "-" * 65)
    print(f"BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN ({NUM_EPOCHS} EPOCHS)")
    print("-" * 65)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        # Chạy 1 epoch train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS
        )

        # Chạy đánh giá trên tập val
        val_loss, val_acc = evaluate_model(
            model, val_loader, criterion, device, epoch, NUM_EPOCHS
        )

        # Cập nhật learning rate
        scheduler.step()
        epoch_duration = time.time() - epoch_start

        # In kết quả tổng hợp của epoch
        print(f"==> Epoch {epoch:02d}/{NUM_EPOCHS:02d} ({epoch_duration:.1f}s) "
              f"| Train Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.4f} - Acc: {val_acc:.2f}%")

        # Lưu checkpoint nếu đạt accuracy cao nhất
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes_meta": meta
            }, CHECKPOINT_PATH)
            print(f"    [KỶ LỤC MỚI] Đã lưu model tốt nhất vào '{CHECKPOINT_PATH}' (Val Acc: {val_acc:.2f}%)")

    total_training_time = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"HUẤN LUYỆN HOÀN TẤT SAU {total_training_time / 60:.2f} PHÚT!")
    print(f"    - Độ chính xác cao nhất trên tập Val: {best_val_accuracy:.2f}%")
    print(f"    - File trọng số đã lưu: {CHECKPOINT_PATH}")
    print("=" * 65)

    # 7. Chạy thử nghiệm dự đoán ngẫu nhiên vài ảnh
    demo_inference(model, val_dataset, idx_to_char, device, num_tests=8)


if __name__ == "__main__":
    main()
