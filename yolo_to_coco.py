#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_to_coco.py  ——  用训练好的 YOLO 模型在 val 图上预测,写成 COCO det json

产出与 to_coco.py 同构(images/annotations/categories,3 核心类名 +
每框 score),从而能直接喂 badcase.py 与 VLM / 真值 同口径对照。

需要 ultralytics(在 5090 上):pip install ultralytics

用法:
  python yolo_to_coco.py --weights runs/detect/train/weights/best.pt \
      --images val_imgs/ --out det_pred_coco.json --conf 0.25 --imgsz 1280
"""

import argparse
import json
import os

from PIL import Image
from badcase import CORE_CLASSES   # ["pedestrian","vehicle","bicycle"]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    ap = argparse.ArgumentParser(description="Trained YOLO -> COCO predictions for badcase.py")
    ap.add_argument("--weights", required=True, help="best.pt")
    ap.add_argument("--images", required=True, help="val 图片目录")
    ap.add_argument("--out", required=True, help="输出 COCO json")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--imgsz", type=int, default=1280, help="推理分辨率(航拍小目标建议 1280+)")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    files = sorted(f for f in os.listdir(args.images) if f.lower().endswith(IMG_EXTS))
    cat_id = {c: i + 1 for i, c in enumerate(CORE_CLASSES)}   # 1..3
    images, anns = [], []
    aid = 1
    for iid, fn in enumerate(files, 1):
        path = os.path.join(args.images, fn)
        with Image.open(path) as im:
            W, H = im.size
        images.append({"id": iid, "file_name": fn, "width": W, "height": H})
        res = model.predict(path, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        for b in res.boxes:
            ci = int(b.cls)
            name = CORE_CLASSES[ci] if ci < len(CORE_CLASSES) else str(ci)
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            anns.append({"id": aid, "image_id": iid, "category_id": cat_id[name],
                         "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0,
                         "score": float(b.conf)})
            aid += 1

    cats = [{"id": i + 1, "name": c} for i, c in enumerate(CORE_CLASSES)]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": anns, "categories": cats}, f)
    print(f"[yolo_to_coco] images={len(images)}  dets={len(anns)}  -> {args.out}")


if __name__ == "__main__":
    main()
