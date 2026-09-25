#!/usr/bin/env python3
"""
convert_to_memmap.py: Chuyển dataset từ HDF5 (.h5, nén LZF) sang .npy memory-map (không nén).

LÝ DO (đây là nguyên nhân chính khiến train chậm):
- File .h5 hiện tại dùng nén LZF với chunk = 2000 ảnh (~24MB mỗi chunk).
- Khi DataLoader shuffle đọc 1 ảnh bất kỳ, h5py phải GIẢI NÉN CẢ CHUNK 24MB để lấy 1 ảnh (12KB).
  -> Mỗi ảnh tốn ~40ms chỉ riêng khâu đọc, GPU ngồi chơi chờ data.
- File .npy (memory-map) không nén, đọc ngẫu nhiên 1 ảnh gần như tức thì (~0.05ms).

Cách dùng:  python convert_to_memmap.py
Sau khi chạy xong, train.py sẽ tự đọc file .npy thay vì .h5.
"""
import os
import h5py
import numpy as np

DATA_DIR = "./dataset"
SPLITS = ["dataset_train", "dataset_val"]

for name in SPLITS:
    src = os.path.join(DATA_DIR, name + ".h5")
    img_dst = os.path.join(DATA_DIR, name + "_images.npy")
    lbl_dst = os.path.join(DATA_DIR, name + "_labels.npy")

    print(f"[*] Chuyển {src} -> .npy ...")
    with h5py.File(src, "r") as f:
        imgs = f["images"]
        labels = f["labels"]

        mm = np.lib.format.open_memmap(img_dst, mode="w+", dtype=imgs.dtype, shape=imgs.shape)
        CHUNK = 20000
        for i in range(0, imgs.shape[0], CHUNK):
            mm[i:i + CHUNK] = imgs[i:i + CHUNK]
            print(f"    {min(i + CHUNK, imgs.shape[0]):,}/{imgs.shape[0]:,}", flush=True)
        mm.flush()
        del mm

        np.save(lbl_dst, labels[:])
    print(f"[OK] {name} xong -> {img_dst} | {lbl_dst}")

print("HOÀN TẤT chuyển đổi.")
