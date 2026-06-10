#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_tile_dataset.py —— 把(伪)标注切成"原生分辨率瓦片"训练集(SAHI 式微调)

动机(Day 14):自训练第 2 轮饱和的根因不只在标签,而在**模型吃不进小目标**:
  1) 训练时整图 1920x1080 被压到 imgsz=1280,8px 目标只剩 ~5px;
  2) YOLOv8 默认最细检测头是 P3(stride 8),亚 stride 目标在架构上学不到。
  → 第 1 轮辛苦补回的小目标标签,在"整图 1280 训练"下大部分被浪费。

本脚本把训练分布改成与 SAHI 推理一致的"瓦片原生分辨率":
  - 每张图切 tile×tile(重叠 overlap),框裁剪进瓦片坐标;
  - 可见面积占原框 < --min-vis 的截断框丢弃(避免半截框噪声);
  - 空瓦片按 --keep-empty 比例保留为负样本(背景);
  - --include-full 时整图也作为样本(大目标的学习信号不丢);
  - 副作用是免费的数据放大:300 图 → ~2500+ 训练样本。

配套:训练用 yolov8s-p2.yaml(加 stride-4 P2 头)+ imgsz=640;
推理用 yolo_to_coco.py --tile 640 --imgsz 640(分布与训练完全对齐)。

用法:
  python make_tile_dataset.py --coco refined_coco.json --images train300/ \
      --out ds_tiles --tile 640 --overlap 0.2 --min-vis 0.4 \
      --keep-empty 0.05 --include-full
"""

import argparse
import json
import os
import shutil
from collections import defaultdict

from PIL import Image
from badcase import DEFAULT_LABEL_MAP, CORE_CLASSES
from yolo_to_coco import _make_tiles

CLS_ID = {c: i for i, c in enumerate(CORE_CLASSES)}   # pedestrian:0 vehicle:1 bicycle:2


def clip_box_to_tile(box, tile):
    """box=(x,y,w,h) 全图坐标;tile=(x0,y0,x1,y1)。
    返回 (clipped_x, clipped_y, clipped_w, clipped_h, vis_ratio) —— 瓦片内坐标。"""
    x, y, w, h = box
    x0, y0, x1, y1 = tile
    ix1, iy1 = max(x, x0), max(y, y0)
    ix2, iy2 = min(x + w, x1), min(y + h, y1)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0 or w <= 0 or h <= 0:
        return None
    vis = (iw * ih) / (w * h)
    return (ix1 - x0, iy1 - y0, iw, ih, vis)


def yolo_line(cls_id, x, y, w, h, W, H):
    cx, cy = (x + w / 2) / W, (y + h / 2) / H
    nw, nh = w / W, h / H
    cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
    nw, nh = min(nw, 1.0), min(nh, 1.0)
    return f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def main():
    ap = argparse.ArgumentParser(description="COCO labels -> native-resolution tile YOLO dataset (SAHI-style fine-tuning)")
    ap.add_argument("--coco", required=True, help="COCO 标注(伪/精/真值皆可)")
    ap.add_argument("--images", required=True, help="原图目录")
    ap.add_argument("--out", required=True, help="输出 YOLO 数据集根目录")
    ap.add_argument("--split", default="train")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--min-vis", type=float, default=0.4,
                    help="框在瓦片内可见面积占比低于此值则丢弃(默认 0.4)")
    ap.add_argument("--keep-empty", type=float, default=0.05,
                    help="空瓦片保留比例,作背景负样本(默认 0.05;0=全丢)")
    ap.add_argument("--include-full", action="store_true",
                    help="整图也作为一个训练样本(保大目标)")
    args = ap.parse_args()

    with open(args.coco, "r", encoding="utf-8") as f:
        coco = json.load(f)
    id2name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    anns_by_img = defaultdict(list)
    n_label_drop = 0
    for a in coco.get("annotations", []):
        core = DEFAULT_LABEL_MAP.get(str(id2name.get(a["category_id"], "")).strip().lower())
        if core is None:
            n_label_drop += 1
            continue
        anns_by_img[a["image_id"]].append((CLS_ID[core], a["bbox"]))

    img_dir = os.path.join(args.out, "images", args.split)
    lbl_dir = os.path.join(args.out, "labels", args.split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    keep_every = round(1.0 / args.keep_empty) if args.keep_empty > 0 else 0
    n_imgs = n_tiles = n_kept = n_empty_kept = n_box = n_vis_drop = 0
    empty_seen = 0

    for im in coco.get("images", []):
        src = os.path.join(args.images, im["file_name"])
        if not os.path.exists(src):
            continue
        stem = os.path.splitext(im["file_name"])[0]
        with Image.open(src) as pil:
            pil = pil.convert("RGB")
            W, H = pil.size
            boxes = anns_by_img.get(im["id"], [])

            for ti, tile in enumerate(_make_tiles(W, H, args.tile, args.overlap)):
                n_tiles += 1
                x0, y0, x1, y1 = tile
                tw, th = x1 - x0, y1 - y0
                lines = []
                for cls_id, bbox in boxes:
                    c = clip_box_to_tile(tuple(bbox), tile)
                    if c is None:
                        continue
                    cx, cy, cw, ch, vis = c
                    if vis < args.min_vis or cw < 2 or ch < 2:
                        n_vis_drop += 1
                        continue
                    lines.append(yolo_line(cls_id, cx, cy, cw, ch, tw, th))

                if not lines:
                    empty_seen += 1
                    if not (keep_every and empty_seen % keep_every == 0):
                        continue
                    n_empty_kept += 1

                name = f"{stem}__t{ti}"
                pil.crop(tile).save(os.path.join(img_dir, name + ".jpg"),
                                    quality=90)
                with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
                    f.write("\n".join(lines) + ("\n" if lines else ""))
                n_kept += 1
                n_box += len(lines)

            if args.include_full:
                name = f"{stem}__full"
                dst = os.path.join(img_dir, name + ".jpg")
                if not os.path.exists(dst):
                    try:
                        os.symlink(os.path.abspath(src), dst)
                    except OSError:
                        shutil.copy2(src, dst)
                lines = [yolo_line(cls_id, *bbox, W, H) for cls_id, bbox in boxes]
                with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
                    f.write("\n".join(lines) + ("\n" if lines else ""))
                n_kept += 1
                n_box += len(lines)
        n_imgs += 1

    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\n")
        f.write(f"train: images/{args.split}\n")
        f.write(f"val: images/{args.split}\n")
        f.write(f"nc: {len(CORE_CLASSES)}\n")
        f.write(f"names: {list(CORE_CLASSES)}\n")

    print(f"[tile_dataset] src_images={n_imgs}  tiles_scanned={n_tiles}  "
          f"samples_kept={n_kept} (empty_neg={n_empty_kept})  boxes={n_box}")
    print(f"               dropped: non-core={n_label_drop}  low-vis/tiny={n_vis_drop}")
    print(f"[OK] dataset -> {args.out}")


if __name__ == "__main__":
    main()
