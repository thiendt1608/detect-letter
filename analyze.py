#!/usr/bin/env python3
"""
analyze.py: Phân tích model đã train xong (best_model.pth) mà KHÔNG cần train lại.

Dùng khi m đã có sẵn file trọng số -> vẽ ma trận nhầm lẫn (confusion matrix)
để xem model hay nhầm kí tự nào với kí tự nào.

Cách dùng:  python analyze.py
"""
import json
import torch
from torch.utils.data import DataLoader

import train  # tái sử dụng config, dataset, model và hàm vẽ từ train.py


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(train.CLASSES_JSON_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    num_classes = meta["num_classes"]
    classes = meta["classes"]
    channels = meta.get("channels", 3)

    if not __import__("os").path.exists(train.CHECKPOINT_PATH):
        print(f"[X] Không tìm thấy '{train.CHECKPOINT_PATH}'. Chạy train.py trước nha!")
        return

    ckpt = torch.load(train.CHECKPOINT_PATH, map_location=device)
    model = train.CharacterClassifierCNN(num_classes, channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[*] Đã nạp model từ '{train.CHECKPOINT_PATH}' (Val Acc: {ckpt.get('val_acc', '?'):.2f}%)")

    if channels == 1:
        mean, std = [0.5], [0.5]
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    val_dataset = train.CharacterDataset(train.VAL_IMG_PATH, train.VAL_LBL_PATH, mean=mean, std=std)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False,
                            num_workers=4, pin_memory=True)

    cm = train.compute_confusion_matrix(model, val_loader, device, num_classes)
    train.plot_confusion_matrix(cm, classes, train.CONFUSION_PATH, normalize=True)


if __name__ == "__main__":
    main()
