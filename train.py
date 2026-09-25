#!/usr/bin/env python3
"""
train.py: File huấn luyện mạng nơ-ron (Convolutional Neural Network - CNN)
nhận diện kí tự đơn lẻ (62 class: 0-9, A-Z, a-z) từ dataset 2M mẫu (file .npy memory-map).

Được chú thích cực kì chi tiết từ A-Z để m dễ dàng chỉnh sửa, thay đổi kiến trúc
hoặc tinh chỉnh hyperparameters (learning rate, batch size, số epoch,...).
"""

import argparse
import os
import sys
import json
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ==============================================================================
# 1. KHU VỰC CẤU HÌNH (HYPERPARAMETERS) - M TINH CHỈNH Ở ĐÂY NHA!
# ==============================================================================
# Đường dẫn dữ liệu (file .npy memory-map, không nén - đọc nhanh hơn .h5 rất nhiều)
DATA_DIR = "./dataset"
TRAIN_IMG_PATH = os.path.join(DATA_DIR, "dataset_train_images.npy")
TRAIN_LBL_PATH = os.path.join(DATA_DIR, "dataset_train_labels.npy")
VAL_IMG_PATH = os.path.join(DATA_DIR, "dataset_val_images.npy")
VAL_LBL_PATH = os.path.join(DATA_DIR, "dataset_val_labels.npy")
CLASSES_JSON_PATH = os.path.join(DATA_DIR, "classes.json")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "best_model.pth")
HISTORY_PATH = os.path.join(DATA_DIR, "training_history.json")   # Lịch sử loss/acc từng epoch
CURVES_PATH = os.path.join(DATA_DIR, "training_curves.png")      # Biểu đồ loss/accuracy
CONFUSION_PATH = os.path.join(DATA_DIR, "confusion_matrix.png")  # Ma trận nhầm lẫn

# Tham số huấn luyện (m có thể tăng giảm tuỳ ý)
BATCH_SIZE = 512          # Số ảnh nạp vào model mỗi lượt (512 chạy mượt trên GPU 12GB)
LEARNING_RATE = 1e-3      # Tốc độ học (0.001 là chuẩn, nếu thấy loss dao động mạnh thì giảm về 3e-4)
NUM_EPOCHS = 10           # Số lần lặp qua toàn bộ dataset (với 1.8M mẫu thì 5-10 epoch là acc > 95% rồi)
NUM_WORKERS = 4           # Số luồng CPU đọc data song song. Trên Windows mỗi worker là 1 tiến trình riêng (ngốn ~1GB RAM), nên để 4 là đủ (data giờ đọc nhanh rồi, không cần 8)
WEIGHT_DECAY = 1e-4       # Hệ số phạt L2 regularization để chống overfitting (học vẹt)
THROTTLE_SECONDS = 0.04   # Delay giữa các batch để giới hạn GPU ~70-80% (0 = chạy hết công suất). Tăng lên nếu muốn GPU thấp hơn.
EARLY_STOPPING_PATIENCE = 3  # Early Stopping: dừng sớm nếu val_acc không cải thiện trong N epoch liên tiếp (0 = tắt)


# ==============================================================================
# 2. CHỌN THIẾT BỊ TÍNH TOÁN (GPU APPLE SILICON / CUDA / CPU)
# ==============================================================================
def get_computing_device() -> torch.device:
    """Tự động chọn GPU: Apple Silicon (MPS) hoặc NVIDIA (CUDA), fallback CPU."""
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
# 3. CLASS NẠP DỮ LIỆU TỪ FILE .NPY (MEMORY-MAP, KHÔNG NÉN)
# ==============================================================================
class CharacterDataset(Dataset):
    """
    Dataset đọc ảnh từ file .npy mở ở chế độ memory-map (mmap).
    KHÁC VỚI BẢN CŨ (đọc .h5 nén LZF):
    - File .h5 nén LZF với chunk 2000 ảnh => đọc 1 ảnh phải giải nén cả chunk (~24MB) -> cực chậm.
    - File .npy không nén => đọc ngẫu nhiên 1 ảnh gần như tức thì, hệ điều hành tự cache vào RAM.
    Mỗi worker (tiến trình) tự mở 1 mmap riêng -> không đụng file pointer, không crash.
    """
    def __init__(self, images_path: str, labels_path: str,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"Không tìm thấy file .npy. M chạy 'python convert_to_memmap.py' trước nha!\n"
                f"  - {images_path}\n  - {labels_path}")

        self.images_path = images_path
        self.labels_path = labels_path
        self.images = None   # Mở lazy trong _lazy_init (vì DataLoader dùng multiprocessing)
        self.labels = None

        # Đọc nhanh metadata (mở memmap không load data, chỉ lấy shape rồi đóng)
        img = np.load(images_path, mmap_mode="r")
        self.total_samples = img.shape[0]
        self.channels = img.shape[-1]
        del img

        # Chuẩn hoá gộp 1 lần: out = (x/255 - mean) / std = x * scale + shift
        # scale/shift có shape (C, 1, 1) để broadcast theo từng kênh màu.
        mean_t = torch.tensor(mean, dtype=torch.float32).reshape(-1, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).reshape(-1, 1, 1)
        self.scale = (1.0 / 255.0) / std_t
        self.shift = -mean_t / std_t

    def _lazy_init(self):
        """Mỗi worker process mở 1 mmap riêng (không đụng nhau, không tốn RAM do dùng chung page cache)."""
        if self.images is None:
            self.images = np.load(self.images_path, mmap_mode="r")   # uint8 (N, H, W, C)
            self.labels = np.load(self.labels_path, mmap_mode="r")   # int64 (N,)

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        self._lazy_init()

        # Đọc mảng numpy uint8 (H, W, C) rồi chuyển thẳng sang tensor float32 (C, H, W)
        img_np = self.images[idx]
        label = int(self.labels[idx])

        img = torch.from_numpy(img_np.copy()).permute(2, 0, 1).to(torch.float32)
        img.mul_(self.scale).add_(self.shift)  # Chuẩn hoá vector hoá, không cần PIL

        return img, label


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

        # Giới hạn tốc độ: cho GPU nghỉ 1 nhịp giữa các batch (giảm nhiệt/tải GPU & SSD)
        if THROTTLE_SECONDS > 0:
            time.sleep(THROTTLE_SECONDS)

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

        # Giới hạn tốc độ: cho GPU nghỉ 1 nhịp giữa các batch
        if THROTTLE_SECONDS > 0:
            time.sleep(THROTTLE_SECONDS)

    return running_loss / total_samples, (correct_predictions / total_samples) * 100


# ==============================================================================
# 6b. VẼ BIỂU ĐỒ VÀ MA TRẬN NHẦM LẪN (TRỰC QUAN HOÁ QUÁ TRÌNH HỌC)
# ==============================================================================
def plot_training_history(history, save_path: str):
    """Vẽ biểu đồ Loss & Accuracy theo từng epoch rồi lưu ra file PNG."""
    import matplotlib
    matplotlib.use("Agg")  # Chế độ không cần cửa sổ hiển thị, chỉ ghi ra file
    import matplotlib.pyplot as plt

    epochs = history["epochs"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss theo epoch"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-o", label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy theo epoch"); ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"    [ĐỒ THỊ] Đã lưu biểu đồ học tập: {save_path}")


def plot_confusion_matrix(cm, classes, save_path: str, normalize: bool = True):
    """Vẽ ma trận nhầm lẫn (confusion matrix) - xem model hay nhầm kí tự nào với kí tự nào."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, label="Tỷ lệ" if normalize else "Số mẫu")

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=5)
    ax.set_yticklabels(classes, fontsize=5)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
    ax.set_title("Confusion Matrix (%)" if normalize else "Confusion Matrix (số mẫu)")
    plt.setp(ax.get_xticklabels(), rotation=90)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"    [ĐỒ THỊ] Đã lưu ma trận nhầm lẫn: {save_path}")


@torch.no_grad()
def compute_confusion_matrix(model, dataloader, device, num_classes):
    """Chạy model qua toàn bộ val set để tính ma trận nhầm lẫn."""
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, labels in tqdm(dataloader, desc="Tính confusion matrix", unit="batch"):
        images = images.to(device, non_blocking=True)
        preds = model(images).argmax(dim=1).cpu().numpy()
        labs = labels.numpy()
        cm += np.bincount(labs * num_classes + preds,
                          minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


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
def parse_args():
    """Xử lý tham số dòng lệnh (--resume, --epochs)."""
    parser = argparse.ArgumentParser(
        description="Huấn luyện CNN nhận diện kí tự đơn lẻ (62 class)."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Tiếp tục huấn luyện từ checkpoint tốt nhất (dataset/best_model.pth)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Ghi đè tổng số epoch (vd: --epochs 20 để huấn luyện tới epoch 20)."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Early Stopping: dừng sớm nếu val_acc không cải thiện N epoch liên tiếp (0 = tắt)."
    )
    return parser.parse_args()


def main():
    print("=" * 65)
    print("BẮT ĐẦU CHƯƠNG TRÌNH HUẤN LUYỆN MODEL NHẬN DIỆN KÍ TỰ")
    print("=" * 65)

    args = parse_args()
    total_epochs = args.epochs if args.epochs is not None else NUM_EPOCHS
    patience = args.patience

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
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # Tự chọn thuật toán conv nhanh nhất cho ảnh 64x64

    # 3. Tham số chuẩn hoá (mean/std) cho ảnh grayscale hoặc RGB
    if channels == 1:
        mean, std = [0.5], [0.5]
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # 4. Khởi tạo Dataset & DataLoader
    print("[*] Đang nạp dataset (memory-map .npy)...")
    train_dataset = CharacterDataset(TRAIN_IMG_PATH, TRAIN_LBL_PATH, mean=mean, std=std)
    val_dataset = CharacterDataset(VAL_IMG_PATH, VAL_LBL_PATH, mean=mean, std=std)
    print(f"Đã kết nối: Train ({len(train_dataset):,} mẫu) | Val ({len(val_dataset):,} mẫu)")

    # DataLoader tự động gom batch và shuffle
    # pin_memory=True + prefetch_factor giúp copy data sang GPU nhanh hơn (chạy song song với GPU)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,                # Trộn ngẫu nhiên data mỗi epoch
        num_workers=NUM_WORKERS,
        persistent_workers=(NUM_WORKERS > 0),
        pin_memory=True,
        prefetch_factor=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=(NUM_WORKERS > 0),
        pin_memory=True,
        prefetch_factor=4
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

    # --- RESUME: nạp lại trọng số + optimizer + lịch sử từ checkpoint tốt nhất ---
    start_epoch = 1
    best_val_accuracy = 0.0
    history = {"epochs": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    scheduler_state = None

    if args.resume:
        if not os.path.exists(CHECKPOINT_PATH):
            print(f"[X] Lỗi: Không tìm thấy checkpoint '{CHECKPOINT_PATH}' để resume!")
            print("    Chạy 'python train.py' (không có --resume) để huấn luyện mới trước.")
            sys.exit(1)

        print(f"[*] Đang nạp checkpoint để resume: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        # Kiểm tra kiến trúc khớp (số class không đổi)
        saved_meta = checkpoint.get("classes_meta", {})
        if saved_meta.get("num_classes", num_classes) != num_classes:
            print(f"[X] Lỗi: Checkpoint dùng {saved_meta.get('num_classes')} class, "
                  f"nhưng dataset hiện tại có {num_classes} class!")
            sys.exit(1)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_accuracy = float(checkpoint.get("val_acc", 0.0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        scheduler_state = checkpoint.get("scheduler_state_dict")

        # Nạp lại lịch sử để vẽ biểu đồ liền mạch (nếu có)
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)

        print(f"[*] Đã khôi phục: epoch {start_epoch - 1} | Val Acc tốt nhất {best_val_accuracy:.2f}%")

    # Lịch giảm dần learning rate (Cosine Annealing) để cuối quá trình model học tinh tế hơn
    # Tạo SAU khi đã nạp optimizer (nếu resume) để base_lrs khớp với learning rate được khôi phục.
    if args.resume and scheduler_state is None:
        # Checkpoint cũ không lưu scheduler state -> LR đã giảm về 0 ở cuối cosine.
        # Reset LR về LEARNING_RATE để model còn học được thêm, rồi chạy cosine mới.
        for g in optimizer.param_groups:
            g["lr"] = LEARNING_RATE
        print("[*] Checkpoint cũ không có scheduler state -> reset LR về LEARNING_RATE.")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    # 6. VÒNG LẬP HUẤN LUYỆN (TRAINING LOOP)
    start_time = time.time()
    no_improve_epochs = 0  # Đếm số epoch liên tiếp val_acc không cải thiện (cho Early Stopping)

    print("\n" + "-" * 65)
    print(f"BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN (EPOCH {start_epoch} -> {total_epochs})")
    print("-" * 65)

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()

        # Chạy 1 epoch train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, total_epochs
        )

        # Chạy đánh giá trên tập val
        val_loss, val_acc = evaluate_model(
            model, val_loader, criterion, device, epoch, total_epochs
        )

        # Cập nhật learning rate
        scheduler.step()
        epoch_duration = time.time() - epoch_start

        # In kết quả tổng hợp của epoch
        print(f"==> Epoch {epoch:02d}/{total_epochs:02d} ({epoch_duration:.1f}s) "
              f"| Train Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% "
              f"| Val Loss: {val_loss:.4f} - Acc: {val_acc:.2f}%")

        # Lưu checkpoint nếu đạt accuracy cao nhất + theo dõi Early Stopping
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            no_improve_epochs = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_acc": val_acc,
                "classes_meta": meta
            }, CHECKPOINT_PATH)
            print(f"    [KỶ LỤC MỚI] Đã lưu model tốt nhất vào '{CHECKPOINT_PATH}' (Val Acc: {val_acc:.2f}%)")
        else:
            no_improve_epochs += 1
            if patience > 0:
                print(f"    [EARLY STOPPING] Không cải thiện {no_improve_epochs}/{patience} epoch liên tiếp")

        # Ghi lại kết quả epoch để cuối chương trình vẽ biểu đồ
        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Dừng sớm nếu val_acc không cải thiện quá patience epoch liên tiếp
        if patience > 0 and no_improve_epochs >= patience:
            print(f"\n[EARLY STOPPING] Dừng sớm ở epoch {epoch}: val_acc không cải thiện {patience} epoch liên tiếp.")
            break

    # Lưu lịch sử training ra JSON + vẽ biểu đồ Loss/Accuracy
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    plot_training_history(history, CURVES_PATH)

    total_training_time = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"HUẤN LUYỆN HOÀN TẤT SAU {total_training_time / 60:.2f} PHÚT!")
    print(f"    - Độ chính xác cao nhất trên tập Val: {best_val_accuracy:.2f}%")
    print(f"    - File trọng số đã lưu: {CHECKPOINT_PATH}")
    print("=" * 65)

    # 7. Chạy thử nghiệm dự đoán ngẫu nhiên vài ảnh
    demo_inference(model, val_dataset, idx_to_char, device, num_tests=8)

    # 8. Vẽ ma trận nhầm lẫn để xem model hay nhầm kí tự nào
    cm = compute_confusion_matrix(model, val_loader, device, num_classes)
    plot_confusion_matrix(cm, meta["classes"], CONFUSION_PATH, normalize=True)


if __name__ == "__main__":
    main()
