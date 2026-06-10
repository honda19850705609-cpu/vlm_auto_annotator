#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coco_to_yolo.py  ——  把 COCO 检测 json 转成 YOLO(ultralytics)训练格式

用于"蒸馏"实验:同一套转换既能处理 **VLM 伪标注**,也能处理 **VisDrone 真值**,
统一映射到 3 个核心类(pedestrian/vehicle/bicycle),从而公平对比
"伪标注训出的检测器" vs "真值训出的检测器"。

- 类别映射复用 badcase.py 的 DEFAULT_LABEL_MAP(自由词/VisDrone 名 → 核心类),
  映射不到核心类的标注直接丢。
- 图像宽高**从图片文件实读**(PIL),不依赖 coco 里的 width/height(可能为 0)。
- 输出标准 ultralytics 目录:images/<split>/、labels/<split>/、data.yaml。

用法:
  python coco_to_yolo.py --coco pseudo.json --images train_imgs/ --out ds_pseudo --split train
  python coco_to_yolo.py --coco real_gt.json --images train_imgs/ --out ds_real   --split train
"""

import argparse
import json
import os
import shutil
from collections import defaultdict

from PIL import Image
from badcase import DEFAULT_LABEL_MAP, CORE_CLASSES   # pedestrian/vehicle/bicycle

CLS_ID = {c: i for i, c in enumerate(CORE_CLASSES)}    # pedestrian:0 vehicle:1 bicycle:2
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    ap = argparse.ArgumentParser(description="COCO det json -> YOLO training set (3 core classes)")
    ap.add_argument("--coco", required=True, help="COCO json(伪标注或真值)")
    ap.add_argument("--images", required=True, help="实际图片所在目录")
    ap.add_argument("--out", required=True, help="输出数据集根目录")
    ap.add_argument("--split", default="train", help="划分名(images/<split>, labels/<split>)")
    ap.add_argument("--copy", action="store_true", help="复制图片而非软链(默认软链)")
    args = ap.parse_args()

    with open(args.coco, "r", encoding="utf-8") as f:
        coco = json.load(f)
    id2name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    imgs = {im["id"]: im for im in coco.get("images", [])}
    anns_by_img = defaultdict(list)
    for a in coco.get("annotations", []):
        anns_by_img[a["image_id"]].append(a)

    lbl_dir = os.path.join(args.out, "labels", args.split)
    img_dir = os.path.join(args.out, "images", args.split)
    os.makedirs(lbl_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    n_img = n_box = n_drop = n_missing = 0
    for iid, im in imgs.items():
        fname = im["file_name"]
        src = os.path.join(args.images, fname)
        if not os.path.exists(src):
            n_missing += 1
            continue
        with Image.open(src) as p:
            W, H = p.size

        # 把图片放进数据集(软链优先,失败再复制)
        dst = os.path.join(img_dir, fname)
        if not os.path.exists(dst):
            if args.copy:
                shutil.copy2(src, dst)
            else:
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    shutil.copy2(src, dst)

        lines = []
        for a in anns_by_img.get(iid, []):
            raw = id2name.get(a["category_id"], str(a["category_id"]))
            cls = DEFAULT_LABEL_MAP.get(str(raw).strip().lower())
            if cls is None:
                n_drop += 1
                continue
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            nw, nh = w / W, h / H
            # clamp 到 [0,1],YOLO 要求
            cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
            nw, nh = min(nw, 1.0), min(nh, 1.0)
            lines.append(f"{CLS_ID[cls]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_box += 1

        stem = os.path.splitext(fname)[0]
        with open(os.path.join(lbl_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        n_img += 1

    yaml_path = os.path.join(args.out, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(args.out)}\n")
        f.write(f"train: images/{args.split}\n")
        f.write(f"val: images/{args.split}\n")   # 占位;真正评估在 val 上外部用 badcase 做
        f.write(f"nc: {len(CORE_CLASSES)}\n")
        f.write(f"names: {list(CORE_CLASSES)}\n")

    print(f"[coco_to_yolo] images={n_img}  boxes={n_box}  "
          f"dropped(non-core)={n_drop}  missing_img={n_missing}")
    print(f"[OK] dataset -> {args.out}  (data.yaml + labels/{args.split}/ + images/{args.split}/)")


if __name__ == "__main__":
    main()
