#!/usr/bin/env python3
"""
preview.py: Sinh nhanh một ảnh lưới (Grid) trực quan 8x8 hoặc 10x10 gồm các kí tự ngẫu nhiên
với đủ kiểu font, độ nghiêng, nhiễu, nền giấy, gradient để kiểm tra độ đa dạng trước khi cày 1M-2M mẫu.
"""

import os
import glob
import random
import argparse
import numpy as np
import cv2
from generate_dataset import CharacterRenderer, get_class_mappings, load_emnist_raw
from multiprocessing import shared_memory


def generate_preview_grid(num_rows: int = 8, num_cols: int = 8, img_size: int = 64,
                          output_path: str = "preview.png", fonts_dir: str = "./fonts",
                          use_emnist: bool = False):
    chars, char_to_idx, idx_to_char = get_class_mappings(case_insensitive=False)

    font_candidates = glob.glob(os.path.join(fonts_dir, "*.ttf")) + glob.glob(os.path.join(fonts_dir, "*.otf"))
    if not font_candidates:
        print("[!] Đang tải font...")
        from download_fonts import main as dl_fonts
        dl_fonts()
        font_candidates = glob.glob(os.path.join(fonts_dir, "*.ttf")) + glob.glob(os.path.join(fonts_dir, "*.otf"))

    shm_emnist = None
    emnist_shape = None
    emnist_class_indices = None

    if use_emnist:
        raw_emnist_imgs, raw_emnist_lbls = load_emnist_raw("./data/emnist")
        num_classes = len(chars)
        emnist_class_indices = {i: [] for i in range(num_classes)}
        for idx, lbl in enumerate(raw_emnist_lbls):
            if lbl < num_classes:
                emnist_class_indices[lbl].append(idx)

        emnist_shape = raw_emnist_imgs.shape
        shm_emnist = shared_memory.SharedMemory(create=True, size=raw_emnist_imgs.nbytes)
        shm_buf = np.ndarray(emnist_shape, dtype=np.uint8, buffer=shm_emnist.buf)
        shm_buf[:] = raw_emnist_imgs[:]

    renderer = CharacterRenderer(
        font_paths=font_candidates,
        img_size=img_size,
        channels=3,
        shm_name=shm_emnist.name if shm_emnist else None,
        emnist_shape=emnist_shape,
        emnist_class_indices=emnist_class_indices
    )

    grid_h = num_rows * img_size
    grid_w = num_cols * img_size
    grid_img = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    print(f"[*] Đang sinh lưới {num_rows}x{num_cols} ({num_rows * num_cols} mẫu ngẫu nhiên)...")
    for r in range(num_rows):
        for c in range(num_cols):
            # Chọn kí tự ngẫu nhiên
            char_idx = random.randint(0, len(chars) - 1)
            char = idx_to_char[char_idx]

            is_emnist = use_emnist and (random.random() < 0.5)
            sample = renderer.generate_single_sample(char, char_idx, is_emnist=is_emnist)

            y0 = r * img_size
            y1 = y0 + img_size
            x0 = c * img_size
            x1 = x0 + img_size

            grid_img[y0:y1, x0:x1] = sample

    # Chuyển RGB sang BGR để OpenCV lưu đúng màu
    bgr_img = cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR)
    if shm_emnist:
        shm_emnist.close()
        shm_emnist.unlink()
    cv2.imwrite(output_path, bgr_img)
    print(f"[✓] Đã lưu ảnh xem thử vào: {os.path.abspath(output_path)}")
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo ảnh lưới xem trước độ đa dạng của dataset.")
    parser.add_argument("--rows", type=int, default=8, help="Số hàng (mặc định: 8)")
    parser.add_argument("--cols", type=int, default=8, help="Số cột (mặc định: 8)")
    parser.add_argument("--size", type=int, default=64, help="Kích thước mỗi ô ảnh (mặc định: 64)")
    parser.add_argument("--output", type=str, default="preview.png", help="Đường dẫn file ảnh xuất")
    parser.add_argument("--use-emnist", action="store_true", help="Bật mẫu chữ viết tay từ EMNIST")
    args = parser.parse_args()

    generate_preview_grid(args.rows, args.cols, args.size, args.output, use_emnist=args.use_emnist)
