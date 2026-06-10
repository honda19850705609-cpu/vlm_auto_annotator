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


# --- SAHI(切图推理)所需的切图 + 跨块 NMS,内联以免引入重依赖 ---
def _axis_starts(length, tile, overlap):
    tile = min(tile, length)
    stride = max(1, int(round(tile * (1.0 - overlap))))
    if tile >= length:
        return [0], tile
    starts = list(range(0, length - tile + 1, stride)) or [0]
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts, tile


def _make_tiles(W, H, tile, overlap):
    xs, tw = _axis_starts(W, tile, overlap)
    ys, th = _axis_starts(H, tile, overlap)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _nms_per_class(dets, iou_thr=0.5):
    """dets: list of (cls_idx, x1, y1, x2, y2, conf). 按类贪心 NMS。"""
    out = []
    by_cls = {}
    for d in dets:
        by_cls.setdefault(d[0], []).append(d)
    for cls, group in by_cls.items():
        group.sort(key=lambda d: d[5], reverse=True)
        kept = []
        for d in group:
            if all(_iou(d[1:5], k[1:5]) < iou_thr for k in kept):
                kept.append(d)
        out.extend(kept)
    return out


def predict_tiled(model, pil_img, tile, overlap, imgsz, conf, full_image=True):
    """SAHI:切块推理(捞小目标)+ 整图推理(保大目标)融合 → 跨块 NMS。

    full_image=True 时把整图那遍的框也并进来,避免切块切断大目标导致大目标召回下降。
    返回 (cls,x1,y1,x2,y2,conf) 列表。
    """
    W, H = pil_img.size
    dets = []
    # 整图一遍:保住大/中目标
    if full_image:
        r = model.predict(pil_img, conf=conf, imgsz=imgsz, verbose=False)[0]
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            dets.append((int(b.cls), x1, y1, x2, y2, float(b.conf)))
    # 切块若干遍:捞小目标
    for (x0, y0, x1, y1) in _make_tiles(W, H, tile, overlap):
        crop = pil_img.crop((x0, y0, x1, y1))
        r = model.predict(crop, conf=conf, imgsz=imgsz, verbose=False)[0]
        for b in r.boxes:
            bx1, by1, bx2, by2 = [float(v) for v in b.xyxy[0]]
            dets.append((int(b.cls), bx1 + x0, by1 + y0, bx2 + x0, by2 + y0, float(b.conf)))
    return _nms_per_class(dets, 0.5)


def main():
    ap = argparse.ArgumentParser(description="Trained YOLO -> COCO predictions for badcase.py")
    ap.add_argument("--weights", required=True, help="best.pt")
    ap.add_argument("--images", required=True, help="val 图片目录")
    ap.add_argument("--out", required=True, help="输出 COCO json")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--imgsz", type=int, default=1280, help="推理分辨率(航拍小目标建议 1280+)")
    ap.add_argument("--tile", type=int, default=0,
                    help="SAHI 切图推理的块边长(px);0=整图推理(默认)。航拍小目标建议 640")
    ap.add_argument("--overlap", type=float, default=0.2, help="SAHI 切块重叠比例")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    mode = f"SAHI tile={args.tile} overlap={args.overlap}" if args.tile > 0 else "whole-image"
    print(f">>> inference mode: {mode}  imgsz={args.imgsz} conf={args.conf}")

    files = sorted(f for f in os.listdir(args.images) if f.lower().endswith(IMG_EXTS))
    cat_id = {c: i + 1 for i, c in enumerate(CORE_CLASSES)}   # 1..3
    images, anns = [], []
    aid = 1
    for iid, fn in enumerate(files, 1):
        path = os.path.join(args.images, fn)
        with Image.open(path) as im:
            im = im.convert("RGB")
            W, H = im.size
            images.append({"id": iid, "file_name": fn, "width": W, "height": H})
            if args.tile > 0:
                # SAHI:切图推理(cls, x1, y1, x2, y2, conf)
                boxes = predict_tiled(model, im, args.tile, args.overlap, args.imgsz, args.conf)
            else:
                res = model.predict(im, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
                boxes = [(int(b.cls), *[float(v) for v in b.xyxy[0]], float(b.conf))
                         for b in res.boxes]
        for ci, x1, y1, x2, y2, sc in boxes:
            name = CORE_CLASSES[ci] if ci < len(CORE_CLASSES) else str(ci)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            anns.append({"id": aid, "image_id": iid, "category_id": cat_id[name],
                         "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0,
                         "score": float(sc)})
            aid += 1

    cats = [{"id": i + 1, "name": c} for i, c in enumerate(CORE_CLASSES)]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": anns, "categories": cats}, f)
    print(f"[yolo_to_coco] images={len(images)}  dets={len(anns)}  -> {args.out}")


if __name__ == "__main__":
    main()
