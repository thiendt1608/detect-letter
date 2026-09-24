#!/usr/bin/env python3
"""
generate_dataset.py: Bộ sinh dataset kí tự đơn (Single Character & Digit) siêu tốc và siêu đa dạng.
- Hỗ trợ 62 class (0-9, A-Z, a-z) hoặc 36 class (--case-insensitive).
- Sử dụng hàng trăm Google Fonts + system fonts.
- Biến đổi nền cực đa dạng: màu bệt, gradient xoay góc, giấy kẻ ngang (lined), giấy kẻ ô (grid), nhiễu hạt (grain/speckles).
- Co dãn nét chữ (dilation/erosion mô phỏng bút bi, bút dạ, bút chì).
- Pipeline Albumentations (C++): xoay, nghiêng, phối cảnh 3D, mờ chuyển động, nhiễu cảm biến, nén JPEG, xước mực.
- Tích hợp chữ viết tay thực tế từ EMNIST qua Shared Memory (zero-copy giữa các tiến trình).
- Lưu dưới dạng HDF5 (.h5) nén LZF cho tốc độ cao và không nghẽn ổ cứng khi cày 1M-2M ảnh, hoặc chuẩn ImageFolder.
"""

import os
import sys
import math
import json
import gzip
import glob
import random
import argparse
import time
from multiprocessing import Pool, cpu_count, shared_memory
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
import h5py
from tqdm import tqdm


# ==========================================
# 1. BẢNG KÍ TỰ & MÃ HOÁ LỚP (CLASSES)
# ==========================================
CHARS_62 = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)] + [chr(ord('a') + i) for i in range(26)]
CHARS_36 = [str(i) for i in range(10)] + [chr(ord('A') + i) for i in range(26)]


def get_class_mappings(case_insensitive: bool = False):
    chars = CHARS_36 if case_insensitive else CHARS_62
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    return chars, char_to_idx, idx_to_char


# ==========================================
# 2. XỬ LÝ & ĐỌC DỮ LIỆU EMNIST (NẾU CÓ)
# ==========================================
def load_emnist_raw(data_dir: str):
    """Đọc dữ liệu EMNIST ByClass từ file .gz."""
    img_file = os.path.join(data_dir, "emnist-byclass-train-images-idx3-ubyte.gz")
    lbl_file = os.path.join(data_dir, "emnist-byclass-train-labels-idx1-ubyte.gz")

    if not os.path.exists(img_file) or not os.path.exists(lbl_file):
        from download_emnist import download_and_extract_emnist
        download_and_extract_emnist(data_dir)

    print("[*] Đang nạp EMNIST vào bộ nhớ...")
    with gzip.open(img_file, "rb") as f:
        _ = f.read(16)  # magic, num, rows, cols
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)

    with gzip.open(lbl_file, "rb") as f:
        _ = f.read(8)  # magic, num
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    test_img_file = os.path.join(data_dir, "emnist-byclass-test-images-idx3-ubyte.gz")
    test_lbl_file = os.path.join(data_dir, "emnist-byclass-test-labels-idx1-ubyte.gz")
    if os.path.exists(test_img_file) and os.path.exists(test_lbl_file):
        with gzip.open(test_img_file, "rb") as f:
            _ = f.read(16)
            test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
        with gzip.open(test_lbl_file, "rb") as f:
            _ = f.read(8)
            test_lbls = np.frombuffer(f.read(), dtype=np.uint8)
        images = np.concatenate([images, test_imgs], axis=0)
        labels = np.concatenate([labels, test_lbls], axis=0)

    print(f"[✓] Đã nạp {len(images):,} mẫu EMNIST viết tay (Train + Test).")
    return images, labels


# ==========================================
# 3. SINH NỀN & HIỆU ỨNG (BACKGROUND & TEXTURE)
# ==========================================
def create_solid_bg(h: int, w: int, channels: int, light_mode: bool) -> np.ndarray:
    if light_mode:
        base = random.randint(200, 255)
        bg = np.full((h, w, channels), base, dtype=np.uint8)
        if channels == 3:
            # Ám màu nhẹ (vàng giấy, xám nhạt, xanh phấn nhẹ)
            tint = np.random.randint(-15, 10, (1, 1, 3))
            bg = np.clip(bg.astype(np.int16) + tint, 0, 255).astype(np.uint8)
    else:
        base = random.randint(15, 60)
        bg = np.full((h, w, channels), base, dtype=np.uint8)
        if channels == 3:
            tint = np.random.randint(-5, 15, (1, 1, 3))
            bg = np.clip(bg.astype(np.int16) + tint, 0, 255).astype(np.uint8)
    return bg


def create_gradient_bg(h: int, w: int, channels: int, light_mode: bool) -> np.ndarray:
    angle_rad = math.radians(random.uniform(0, 360))
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    proj = xx * math.cos(angle_rad) + yy * math.sin(angle_rad)
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)

    if light_mode:
        c1 = np.random.randint(210, 255, size=channels)
        c2 = np.random.randint(180, 240, size=channels)
    else:
        c1 = np.random.randint(10, 50, size=channels)
        c2 = np.random.randint(40, 80, size=channels)

    bg = np.zeros((h, w, channels), dtype=np.float32)
    for c in range(channels):
        bg[:, :, c] = (1.0 - proj) * c1[c] + proj * c2[c]
    return np.clip(bg, 0, 255).astype(np.uint8)


def add_paper_texture(img: np.ndarray, light_mode: bool) -> np.ndarray:
    h, w, c = img.shape
    style = random.choices(["none", "noise", "grid", "lined", "speckles"], weights=[0.25, 0.35, 0.15, 0.15, 0.10])[0]

    if style == "none":
        return img

    res = img.astype(np.int16)

    if style == "noise":
        noise = np.random.randint(-12, 13, img.shape)
        res = np.clip(res + noise, 0, 255).astype(np.uint8)

    elif style == "grid" and light_mode:
        step = random.randint(10, 16)
        line_color = np.array([195, 205, 215] if c == 3 else [205], dtype=np.int16)
        for y in range(0, h, step):
            res[y, :, :] = np.clip(res[y, :, :] * 0.8 + line_color * 0.2, 0, 255)
        for x in range(0, w, step):
            res[:, x, :] = np.clip(res[:, x, :] * 0.8 + line_color * 0.2, 0, 255)
        res = res.astype(np.uint8)

    elif style == "lined" and light_mode:
        step = random.randint(14, 20)
        line_color = np.array([190, 200, 220] if c == 3 else [200], dtype=np.int16)
        for y in range(step // 2, h, step):
            res[y, :, :] = np.clip(res[y, :, :] * 0.75 + line_color * 0.25, 0, 255)
        res = res.astype(np.uint8)

    elif style == "speckles":
        num_dots = random.randint(2, 6)
        for _ in range(num_dots):
            ry, rx = random.randint(0, h - 1), random.randint(0, w - 1)
            dot_val = random.randint(30, 80) if light_mode else random.randint(180, 230)
            res[ry, rx, :] = dot_val
        res = res.astype(np.uint8)

    return res


# ==========================================
# 4. ALBUMENTATIONS PIPELINE (C++ ACCELERATED)
# ==========================================
def get_augmentation_pipeline(img_size: int):
    return A.Compose([
        # Biến đổi hình học & phối cảnh
        A.Affine(scale=(0.88, 1.12), rotate=(-22, 22), shear=(-12, 12), p=0.85),
        A.Perspective(scale=(0.03, 0.07), p=0.45),
        A.ElasticTransform(alpha=1, sigma=20, p=0.20),

        # Mờ quang học & mờ rung tay
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=0.6),
            A.MotionBlur(blur_limit=(3, 5), p=0.4),
        ], p=0.40),

        # Nhiễu cảm biến camera
        A.GaussNoise(std_range=(0.02, 0.10), p=0.50),
        A.MultiplicativeNoise(multiplier=(0.92, 1.08), p=0.30),

        # Đứt nét, xước mực hoặc bụi
        A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(2, 4), hole_width_range=(2, 4), p=0.30),

        # Nén ảnh JPEG (giả lập camera/web chất lượng thấp)
        A.ImageCompression(quality_range=(55, 95), p=0.45),

        # Chỉnh sáng tối & tương phản nhẹ
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.40),
    ])


# ==========================================
# 5. RENDER CHỮ TỪ FONT HOẶC EMNIST
# ==========================================
class CharacterRenderer:
    def __init__(self, font_paths: List[str], img_size: int = 64, channels: int = 3,
                 shm_name: Optional[str] = None, emnist_shape: Optional[Tuple[int, ...]] = None,
                 emnist_class_indices: Optional[Dict[int, List[int]]] = None):
        self.font_paths = font_paths
        self.img_size = img_size
        self.channels = channels
        self.augmentor = get_augmentation_pipeline(img_size)

        # Caching font objects: (font_path, font_size) -> ImageFont
        self._font_cache = {}

        # Thiết lập SharedMemory cho EMNIST nếu có
        self.shm_emnist = None
        self.emnist_images = None
        self.emnist_class_indices = emnist_class_indices
        if shm_name and emnist_shape:
            self.shm_emnist = shared_memory.SharedMemory(name=shm_name)
            self.emnist_images = np.ndarray(emnist_shape, dtype=np.uint8, buffer=self.shm_emnist.buf)

    def get_font(self, font_path: str, size: int) -> ImageFont.FreeTypeFont:
        key = (font_path, size)
        if key not in self._font_cache:
            try:
                self._font_cache[key] = ImageFont.truetype(font_path, size=size)
            except Exception:
                self._font_cache[key] = ImageFont.load_default()
        return self._font_cache[key]

    def render_font_char(self, char: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Tạo mask chữ từ TTF font với kích thước, căn chỉnh và độ dày nét ngẫu nhiên."""
        font_path = random.choice(self.font_paths)
        # Font size chiếm từ 45% đến 85% chiều cao canvas
        font_size = random.randint(int(self.img_size * 0.45), int(self.img_size * 0.85))
        font = self.get_font(font_path, font_size)

        mask_img = Image.new("L", (self.img_size, self.img_size), color=0)
        draw = ImageDraw.Draw(mask_img)

        # Tính bbox để căn giữa
        bbox = draw.textbbox((0, 0), char, font=font)
        tw = max(1, bbox[2] - bbox[0])
        th = max(1, bbox[3] - bbox[1])

        # Random vị trí lệch tâm nhẹ (+-10% kích thước)
        jitter_x = random.randint(-int(self.img_size * 0.08), int(self.img_size * 0.08))
        jitter_y = random.randint(-int(self.img_size * 0.08), int(self.img_size * 0.08))

        tx = (self.img_size - tw) // 2 - bbox[0] + jitter_x
        ty = (self.img_size - th) // 2 - bbox[1] + jitter_y

        draw.text((tx, ty), char, fill=255, font=font)
        mask = np.array(mask_img)

        # Biến đổi nét chữ (Morphological): Đậm (Dilation) hoặc Mảnh (Erosion)
        morph_choice = random.choices(["none", "dilate", "erode"], weights=[0.60, 0.25, 0.15])[0]
        if morph_choice == "dilate":
            k_size = random.choice([2, 3])
            kernel = np.ones((k_size, k_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        elif morph_choice == "erode":
            kernel = np.ones((2, 2), np.uint8)
            eroded = cv2.erode(mask, kernel, iterations=1)
            if np.count_nonzero(eroded) > 20:  # Tránh làm mất nét hoàn toàn
                mask = eroded

        return mask

    def render_emnist_char(self, class_idx: int) -> np.ndarray:
        """Lấy một mẫu chữ viết tay từ EMNIST và resize/căn vị trí tự nhiên."""
        indices = self.emnist_class_indices.get(class_idx, [])
        if not indices or self.emnist_images is None:
            return None

        sample_idx = random.choice(indices)
        raw_emnist = self.emnist_images[sample_idx]  # 28x28

        # Khôi phục hướng đứng (EMNIST gốc bị xoay và lật ngang)
        upright = np.rot90(np.fliplr(raw_emnist))

        # Random kích thước hiển thị trong canvas (50% - 85% canvas size)
        target_char_size = random.randint(int(self.img_size * 0.50), int(self.img_size * 0.85))
        resized = cv2.resize(upright, (target_char_size, target_char_size), interpolation=cv2.INTER_LINEAR)

        # Tạo canvas và dán vào giữa kèm jitter
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        jx = random.randint(-int(self.img_size * 0.08), int(self.img_size * 0.08))
        jy = random.randint(-int(self.img_size * 0.08), int(self.img_size * 0.08))
        x0 = max(0, min(self.img_size - target_char_size, (self.img_size - target_char_size) // 2 + jx))
        y0 = max(0, min(self.img_size - target_char_size, (self.img_size - target_char_size) // 2 + jy))

        mask[y0:y0 + target_char_size, x0:x0 + target_char_size] = resized
        return mask

    def generate_single_sample(self, char: str, class_idx: int, is_emnist: bool = False) -> np.ndarray:
        """Tạo 1 ảnh hoàn chỉnh: Nền -> Vẽ chữ -> Trộn màu -> Augmentation."""
        h, w, c = self.img_size, self.img_size, self.channels

        # 1. Quyết định chế độ sáng/tối (75% nền sáng chữ tối, 25% nền tối chữ sáng)
        light_mode = random.random() < 0.75

        # 2. Tạo nền
        if random.random() < 0.40:
            bg = create_gradient_bg(h, w, c, light_mode)
        else:
            bg = create_solid_bg(h, w, c, light_mode)
        bg = add_paper_texture(bg, light_mode)

        # 3. Tạo mask chữ
        mask = None
        if is_emnist and self.emnist_class_indices is not None:
            mask = self.render_emnist_char(class_idx)

        if mask is None:
            mask = self.render_font_char(char)

        # 4. Chọn màu chữ đảm bảo độ tương phản cao với nền
        if light_mode:
            # Chữ tối (đen, than, xanh thẫm, nâu đậm, đỏ đô)
            if c == 3:
                text_color = random.choice([
                    np.array([15, 15, 15]),       # Đen tuyền
                    np.array([25, 30, 45]),       # Xanh tím than bút bi
                    np.array([40, 25, 20]),       # Nâu đất
                    np.array([55, 15, 20]),       # Đỏ đô
                    np.array([20, 40, 25]),       # Xanh rêu
                ])
            else:
                text_color = np.array([random.randint(5, 45)])
        else:
            # Chữ sáng (trắng, vàng kem, cyan nhạt, cam nhạt)
            if c == 3:
                text_color = random.choice([
                    np.array([245, 245, 245]),   # Trắng phấn
                    np.array([255, 240, 180]),   # Vàng phấn
                    np.array([190, 245, 255]),   # Xanh phấn
                    np.array([255, 200, 190]),   # Cam nhạt
                ])
            else:
                text_color = np.array([random.randint(215, 255)])

        # 5. Hoà trộn nét chữ lên nền bằng Alpha Blending
        alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
        blended = (1.0 - alpha) * bg.astype(np.float32) + alpha * text_color.astype(np.float32)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # 6. Chạy qua pipeline Albumentations
        augmented = self.augmentor(image=blended)["image"]
        return augmented


# ==========================================
# 6. WORKER PROCESS CHO MULTIPROCESSING
# ==========================================
_global_renderer = None


def _init_worker(font_paths: List[str], img_size: int, channels: int,
                 shm_name: Optional[str], emnist_shape: Optional[Tuple[int, ...]],
                 emnist_class_indices: Optional[Dict[int, List[int]]]):
    global _global_renderer
    _global_renderer = CharacterRenderer(
        font_paths=font_paths,
        img_size=img_size,
        channels=channels,
        shm_name=shm_name,
        emnist_shape=emnist_shape,
        emnist_class_indices=emnist_class_indices
    )


def _worker_generate_chunk(tasks_chunk: List[Tuple[str, int, bool]]) -> Tuple[np.ndarray, np.ndarray]:
    """Sinh một mảng ảnh & nhãn cho chunk tasks."""
    global _global_renderer
    n = len(tasks_chunk)
    h = _global_renderer.img_size
    w = _global_renderer.img_size
    c = _global_renderer.channels

    imgs = np.empty((n, h, w, c), dtype=np.uint8)
    lbls = np.empty(n, dtype=np.int64)

    for i, (char, class_idx, is_emnist) in enumerate(tasks_chunk):
        imgs[i] = _global_renderer.generate_single_sample(char, class_idx, is_emnist)
        lbls[i] = class_idx

    return imgs, lbls


def _worker_write_images_chunk(args_tuple):
    """Ghi trực tiếp ảnh ra đĩa theo cấu trúc ImageFolder."""
    tasks_chunk, base_out_dir, start_idx = args_tuple
    global _global_renderer
    manifest_rows = []

    for i, (char, class_idx, is_emnist, split_name, class_name) in enumerate(tasks_chunk):
        img = _global_renderer.generate_single_sample(char, class_idx, is_emnist)
        global_id = start_idx + i
        rel_path = os.path.join(split_name, class_name, f"{global_id:08d}.jpg")
        full_path = os.path.join(base_out_dir, rel_path)

        # Chuyển RGB sang BGR để cv2.imwrite ghi đúng màu
        if _global_renderer.channels == 3:
            write_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            write_img = img

        cv2.imwrite(full_path, write_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        manifest_rows.append(f"{rel_path},{class_idx}\n")

    return manifest_rows


# ==========================================
# 7. CHƯƠNG TRÌNH ĐIỀU PHỐI CHÍNH (MAIN ENGINE)
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Bộ sinh dataset kí tự đơn 1M - 2M mẫu siêu tốc bằng PIL + Albumentations + Google Fonts + EMNIST.")
    parser.add_argument("--num-samples", type=int, default=10000, help="Tổng số ảnh cần sinh (mặc định 10,000 để thử nghiệm, truyền 2000000 để sinh 2M mẫu).")
    parser.add_argument("--image-size", type=int, default=64, help="Kích thước ảnh vuông (mặc định: 64x64).")
    parser.add_argument("--channels", type=int, default=3, choices=[1, 3], help="Số kênh màu (3: RGB, 1: Grayscale).")
    parser.add_argument("--output-dir", type=str, default="./dataset", help="Thư mục xuất dataset.")
    parser.add_argument("--format", type=str, choices=["h5", "image_folder"], default="h5",
                        help="Định dạng lưu: 'h5' (siêu nhanh, gọn, không hại ổ cứng) hoặc 'image_folder' (file jpg truyền thống).")
    parser.add_argument("--fonts-dir", type=str, default="./fonts", help="Thư mục chứa font .ttf/.otf.")
    parser.add_argument("--use-emnist", action="store_true", help="Bật trộn thêm chữ viết tay thật từ EMNIST.")
    parser.add_argument("--emnist-ratio", type=float, default=0.35, help="Tỷ lệ mẫu chữ viết tay EMNIST (0.0 đến 1.0, mặc định: 0.35).")
    parser.add_argument("--val-split", type=float, default=0.1, help="Tỷ lệ tập validation (mặc định: 0.1 tức 10%%).")
    parser.add_argument("--case-insensitive", action="store_true", help="Chỉ phân loại 36 class (0-9, A-Z) thay vì 62 class.")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Số tiến trình CPU song song (mặc định: toàn bộ core).")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Số lượng mẫu mỗi chunk giao cho worker.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    chars, char_to_idx, idx_to_char = get_class_mappings(args.case_insensitive)
    num_classes = len(chars)

    # Lưu metadata lớp
    classes_meta = {
        "num_classes": num_classes,
        "case_insensitive": args.case_insensitive,
        "image_size": args.image_size,
        "channels": args.channels,
        "classes": chars,
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
    }
    with open(os.path.join(args.output_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(classes_meta, f, indent=2, ensure_ascii=False)

    print("=" * 65)
    print(f"[*] BẮT ĐẦU SINH DATASET KÍ TỰ")
    print(f"    - Tổng số mẫu: {args.num_samples:,}")
    print(f"    - Số lớp (classes): {num_classes}")
    print(f"    - Kích thước ảnh: {args.image_size}x{args.image_size}x{args.channels}")
    print(f"    - Định dạng xuất: {args.format}")
    print(f"    - Số tiến trình CPU (workers): {args.workers}")
    print(f"    - Tích hợp EMNIST viết tay: {'Bật' if args.use_emnist else 'Tắt'}")
    print("=" * 65)

    # 1. Kiểm tra fonts
    if not os.path.exists(args.fonts_dir) or len(glob.glob(os.path.join(args.fonts_dir, "*.*"))) == 0:
        print("[!] Không tìm thấy font trong './fonts'. Đang tự động gọi download_fonts.py...")
        from download_fonts import main as dl_fonts_main
        dl_fonts_main()

    font_candidates = glob.glob(os.path.join(args.fonts_dir, "*.ttf")) + glob.glob(os.path.join(args.fonts_dir, "*.otf"))
    if not font_candidates:
        print("[X] Lỗi: Không tìm thấy font hợp lệ nào!")
        sys.exit(1)
    print(f"[✓] Đã sẵn sàng {len(font_candidates)} font đa dạng.")

    # 2. Xử lý EMNIST (nếu bật)
    shm_emnist = None
    emnist_shape = None
    emnist_class_indices = None

    if args.use_emnist:
        raw_emnist_imgs, raw_emnist_lbls = load_emnist_raw("./data/emnist")
        # Phân loại index theo class
        emnist_class_indices = {i: [] for i in range(num_classes)}
        for idx, lbl in enumerate(raw_emnist_lbls):
            target_class = lbl
            if args.case_insensitive and target_class >= 36:
                target_class = target_class - 26  # Map lowercase 'a'-'z' -> 'A'-'Z'
            if target_class < num_classes:
                emnist_class_indices[target_class].append(idx)

        # Đưa vào SharedMemory để các worker dùng chung không tốn RAM
        emnist_shape = raw_emnist_imgs.shape
        shm_emnist = shared_memory.SharedMemory(create=True, size=raw_emnist_imgs.nbytes)
        shm_buf = np.ndarray(emnist_shape, dtype=np.uint8, buffer=shm_emnist.buf)
        shm_buf[:] = raw_emnist_imgs[:]
        print(f"[✓] Đã tạo SharedMemory cho {len(raw_emnist_imgs):,} ảnh EMNIST ({raw_emnist_imgs.nbytes / (1024*1024):.1f} MB).")

    try:
        # 3. Phân bổ mẫu đều các lớp & chia train/val
        num_val = int(args.num_samples * args.val_split)
        num_train = args.num_samples - num_val

        def build_task_list(total_count: int, split_name: str):
            tasks = []
            base_per_class = total_count // num_classes
            remainder = total_count % num_classes

            for c_idx in range(num_classes):
                c_char = idx_to_char[c_idx]
                c_count = base_per_class + (1 if c_idx < remainder else 0)
                for _ in range(c_count):
                    # Quyết định mẫu này dùng EMNIST hay Font
                    is_emnist = False
                    if args.use_emnist and (random.random() < args.emnist_ratio):
                        if emnist_class_indices.get(c_idx):
                            is_emnist = True
                    # class_name an toàn cho thư mục (phân biệt HOA/thường nếu cần)
                    class_name = f"class_{c_idx:02d}_{c_char}" if c_char.isalnum() else f"class_{c_idx:02d}"
                    tasks.append((c_char, c_idx, is_emnist, split_name, class_name))

            random.shuffle(tasks)
            return tasks

        print("[*] Đang lên danh sách phân bổ mẫu...")
        train_tasks = build_task_list(num_train, "train")
        val_tasks = build_task_list(num_val, "val")
        print(f"[+] Tập Train: {len(train_tasks):,} mẫu | Tập Val: {len(val_tasks):,} mẫu.")

        # 4. Thực thi sinh dữ liệu song song
        t0 = time.time()

        if args.format == "h5":
            train_h5_path = os.path.join(args.output_dir, "dataset_train.h5")
            val_h5_path = os.path.join(args.output_dir, "dataset_val.h5")

            for split_name, task_list, h5_path in [("train", train_tasks, train_h5_path), ("val", val_tasks, val_h5_path)]:
                n_samples = len(task_list)
                if n_samples == 0:
                    continue

                print(f"\n[*] Đang sinh tập {split_name.upper()} ({n_samples:,} mẫu) vào {h5_path}...")
                with h5py.File(h5_path, "w") as h5f:
                    img_dset = h5f.create_dataset(
                        "images",
                        shape=(n_samples, args.image_size, args.image_size, args.channels),
                        dtype=np.uint8,
                        chunks=(min(args.chunk_size, n_samples), args.image_size, args.image_size, args.channels),
                        compression="lzf"
                    )
                    lbl_dset = h5f.create_dataset(
                        "labels",
                        shape=(n_samples,),
                        dtype=np.int64
                    )

                    # Chia tasks thành các chunks
                    chunks = [task_list[i:i + args.chunk_size] for i in range(0, n_samples, args.chunk_size)]
                    simplified_chunks = [[(t[0], t[1], t[2]) for t in chunk] for chunk in chunks]

                    init_args = (
                        font_candidates,
                        args.image_size,
                        args.channels,
                        shm_emnist.name if shm_emnist else None,
                        emnist_shape,
                        emnist_class_indices
                    )

                    with Pool(processes=args.workers, initializer=_init_worker, initargs=init_args) as pool:
                        cursor = 0
                        with tqdm(total=n_samples, desc=f"Generating {split_name}", unit="img") as pbar:
                            for batch_imgs, batch_lbls in pool.imap(_worker_generate_chunk, simplified_chunks):
                                b_size = len(batch_lbls)
                                img_dset[cursor:cursor + b_size] = batch_imgs
                                lbl_dset[cursor:cursor + b_size] = batch_lbls
                                cursor += b_size
                                pbar.update(b_size)

            print(f"[✓] Đã tạo xong file HDF5 train & val!")

        elif args.format == "image_folder":
            # Tạo sẵn cây thư mục
            for split_name, task_list in [("train", train_tasks), ("val", val_tasks)]:
                for c_idx in range(num_classes):
                    c_char = idx_to_char[c_idx]
                    class_name = f"class_{c_idx:02d}_{c_char}"
                    os.makedirs(os.path.join(args.output_dir, split_name, class_name), exist_ok=True)

            for split_name, task_list in [("train", train_tasks), ("val", val_tasks)]:
                n_samples = len(task_list)
                if n_samples == 0:
                    continue

                print(f"\n[*] Đang sinh tập {split_name.upper()} ({n_samples:,} mẫu) ra thư mục...")
                chunks = []
                for i in range(0, n_samples, args.chunk_size):
                    chunk_slice = task_list[i:i + args.chunk_size]
                    chunks.append((chunk_slice, args.output_dir, i))

                init_args = (
                    font_candidates,
                    args.image_size,
                    args.channels,
                    shm_emnist.name if shm_emnist else None,
                    emnist_shape,
                    emnist_class_indices
                )

                manifest_lines = []
                with Pool(processes=args.workers, initializer=_init_worker, initargs=init_args) as pool:
                    with tqdm(total=n_samples, desc=f"Writing {split_name}", unit="img") as pbar:
                        for chunk_rows in pool.imap_unordered(_worker_write_images_chunk, chunks):
                            manifest_lines.extend(chunk_rows)
                            pbar.update(len(chunk_rows))

                # Ghi file manifest
                manifest_file = os.path.join(args.output_dir, f"{split_name}_manifest.csv")
                with open(manifest_file, "w", encoding="utf-8") as mf:
                    mf.write("filepath,label\n")
                    mf.writelines(manifest_lines)

        elapsed = time.time() - t0
        speed = args.num_samples / max(1e-5, elapsed)
        print("\n" + "=" * 65)
        print(f"[🎉] HOÀN TẤT SINH DATASET!")
        print(f"    - Tổng thời gian: {elapsed:.2f} giây ({elapsed / 60:.2f} phút)")
        print(f"    - Tốc độ sinh trung bình: {speed:.1f} ảnh/giây")
        print(f"    - Vị trí dataset: {os.path.abspath(args.output_dir)}")
        print("=" * 65)

    finally:
        # Giải phóng SharedMemory
        if shm_emnist is not None:
            shm_emnist.close()
            shm_emnist.unlink()


if __name__ == "__main__":
    main()
