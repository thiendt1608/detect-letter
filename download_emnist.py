#!/usr/bin/env python3
"""
download_emnist.py: Tải dataset chữ viết tay EMNIST ByClass siêu tốc từ HuggingFace mirror
sử dụng đa luồng (HTTP Range Requests) song song, vượt qua giới hạn băng thông và lỗi 403.
"""

import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

HF_BASE_URL = "https://huggingface.co/datasets/Royc30ne/emnist-byclass/resolve/main/"
TARGET_DIR = "./data/emnist"

FILES_TO_DOWNLOAD = [
    ("emnist-byclass-train-labels-idx1-ubyte.gz", "Labels tập train (~500 KB)"),
    ("emnist-byclass-mapping.txt", "File mapping kí tự (~1 KB)"),
    ("emnist-byclass-train-images-idx3-ubyte.gz", "Ảnh tập train (~360 MB)"),
]


def get_remote_file_size(url: str) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}, method="HEAD")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return int(resp.headers.get("Content-Length", 0))


def download_chunk(url: str, start: int, end: int, dest_file: str, pbar):
    max_retries = 5
    expected_len = end - start + 1
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"}
            )
            with urllib.request.urlopen(req, timeout=40) as resp:
                chunk = resp.read()
                if len(chunk) == expected_len:
                    with open(dest_file, "r+b") as f:
                        f.seek(start)
                        f.write(chunk)
                    pbar.update(len(chunk))
                    return
        except Exception:
            time.sleep(1 + attempt)
    raise RuntimeError(f"Tải chunk {start}-{end} thất bại sau {max_retries} lần thử.")

def is_complete_file(filepath: str) -> bool:
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 1000:
        return False
    if filepath.endswith(".gz"):
        import gzip
        try:
            with gzip.open(filepath, "rb") as f:
                while f.read(10 * 1024 * 1024):
                    pass
            return True
        except Exception:
            return False
    return True


def download_file_multithreaded(url: str, dest_path: str, desc: str, num_threads: int = 8):
    if is_complete_file(dest_path):
        print(f"Đã có sẵn hoàn chỉnh: {os.path.basename(dest_path)}")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    total_size = get_remote_file_size(url)
    part_path = dest_path + ".part"

    if total_size <= 0:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
            f.write(resp.read())
        return

    # Tạo file rỗng đúng kích thước
    with open(part_path, "wb") as f:
        f.truncate(total_size)

    chunk_size = 4 * 1024 * 1024
    ranges = []
    for start in range(0, total_size, chunk_size):
        end = min(start + chunk_size - 1, total_size - 1)
        ranges.append((start, end))

    print(f"[*] Đang tải {desc} ({total_size / (1024*1024):.1f} MB, {len(ranges)} chunks song song)...")
    with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc=os.path.basename(dest_path)) as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(download_chunk, url, start, end, part_path, pbar)
                for start, end in ranges
            ]
            for future in futures:
                future.result()

    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.rename(part_path, dest_path)


def download_and_extract_emnist(dest_dir=TARGET_DIR):
    os.makedirs(dest_dir, exist_ok=True)
    train_img_file = os.path.join(dest_dir, "emnist-byclass-train-images-idx3-ubyte.gz")
    train_lbl_file = os.path.join(dest_dir, "emnist-byclass-train-labels-idx1-ubyte.gz")

    if is_complete_file(train_img_file) and is_complete_file(train_lbl_file):
        print(f"EMNIST dataset đã tồn tại hoàn chỉnh ở: {dest_dir}")
        return dest_dir
    print("[*] Bắt đầu tải EMNIST ByClass từ mirror HuggingFace...")
    for filename, desc in FILES_TO_DOWNLOAD:
        file_url = HF_BASE_URL + filename
        dest_file = os.path.join(dest_dir, filename)
        download_file_multithreaded(file_url, dest_file, desc, num_threads=8)

    print(f"Toàn bộ dữ liệu EMNIST đã sẵn sàng tại: {dest_dir}")
    return dest_dir


if __name__ == "__main__":
    download_and_extract_emnist()
