#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sahi_relabel.py  ——  切图(SAHI)自训练:用检测器补全小目标标签

自训练的核心一步。诊断:伪标注训出的检测器召回缺口全在小目标,根因是 VLM 伪标注
本身漏标了小目标 → 漏标区域成了"背景"负监督 → 检测器被训成忽略小目标。

解法:用训好的检测器对 train 集做**切图推理(SAHI: Slicing Aided Hyper Inference)**
——把图切成带重叠的小块 + 整图,各自 predict 后拼回全图 + 全局 NMS。切块让检测器
在 1280 分辨率下看 640 的块,小目标等效放大 ~2×,能捞回整图(及 VLM)漏掉的小目标。
再与 VLM 伪标注**融合(并集 + NMS)**,得到"补全后"的标签,用于下一轮重训。

纯检测器 + 切图,零人工、零新模型。只依赖 ultralytics + PIL。

用法:
  python sahi_relabel.py --weights runs/pseudo/weights/best.pt \
      --images train300/ --vlm-coco pseudo_coco.json \
      --out refined_coco.json --tile 640 --overlap 0.2 --imgsz 1280 --conf 0.25
"""

import argparse
import json
import os
from collections import defaultdict

from PIL import Image
from badcase import CORE_CLASSES, DEFAULT_LABEL_MAP

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def axis_starts(length, tile, overlap):
    tile = min(tile, length)
    stride = max(1, int(round(tile * (1 - overlap))))
    if tile >= length:
        return [0], tile
    s = list(range(0, length - tile + 1, stride)) or [0]
    if s[-1] != length - tile:
        s.append(length - tile)
    return s, tile


def make_tiles(W, H, tile, overlap):
    xs, tw = axis_starts(W, tile, overlap)
    ys, th = axis_starts(H, tile, overlap)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def nms_per_label(dets, thr):
    """按 label 分组贪心 NMS。dets: [{label, bbox[xyxy], score}]。"""
    by = defaultdict(list)
    for d in dets:
        by[d["label"]].append(d)
    out = []
    for _lab, g in by.items():
        g.sort(key=lambda d: -d["score"])
        keep = []
        for d in g:
            if all(iou_xyxy(d["bbox"], k["bbox"]) < thr for k in keep):
                keep.append(d)
        out.extend(keep)
    return out


def load_vlm_by_file(vlm_coco):
    """读 VLM 伪标注 coco,按文件名归核心类,返回 {fname:[{label,bbox xyxy,score}]}。"""
    vc = json.load(open(vlm_coco, "r", encoding="utf-8"))
    id2name = {c["id"]: c["name"] for c in vc.get("categories", [])}
    imgid2f = {im["id"]: im["file_name"] for im in vc.get("images", [])}
    out = defaultdict(list)
    for a in vc.get("annotations", []):
        f = imgid2f.get(a["image_id"])
        core = DEFAULT_LABEL_MAP.get(str(id2name.get(a["category_id"], "")).strip().lower())
        if f is None or core is None:
            continue
        x, y, w, h = a["bbox"]
        out[f].append({"label": core, "bbox": [x, y, x + w, y + h],
                       "score": float(a.get("score", 0.5))})
    return out


def main():
    ap = argparse.ArgumentParser(description="SAHI tiled self-relabeling for self-training")
    ap.add_argument("--weights", required=True, help="trained YOLO best.pt (D0)")
    ap.add_argument("--images", required=True, help="train 图目录")
    ap.add_argument("--out", required=True, help="输出补全后的 COCO")
    ap.add_argument("--vlm-coco", default=None, help="可选:与 VLM 伪标注融合(并集+NMS)")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--nms-iou", type=float, default=0.6)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    vlm_by_file = load_vlm_by_file(args.vlm_coco) if args.vlm_coco else {}

    files = sorted(f for f in os.listdir(args.images) if f.lower().endswith(IMG_EXTS))
    cat_id = {c: i + 1 for i, c in enumerate(CORE_CLASSES)}
    images, anns = [], []
    aid = 1
    n_det = n_vlm = 0
    for iid, fn in enumerate(files, 1):
        path = os.path.join(args.images, fn)
        with Image.open(path) as im:
            W, H = im.size
            img = im.convert("RGB")
        images.append({"id": iid, "file_name": fn, "width": W, "height": H})

        # 切块 + 整图(整图保大目标,切块抠小目标)
        dets = []
        regions = make_tiles(W, H, args.tile, args.overlap) + [(0, 0, W, H)]
        for (x0, y0, x1, y1) in regions:
            crop = img.crop((x0, y0, x1, y1))
            r = model.predict(crop, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            for b in r.boxes:
                ci = int(b.cls)
                if ci >= len(CORE_CLASSES):
                    continue
                bx1, by1, bx2, by2 = [float(v) for v in b.xyxy[0]]
                dets.append({"label": CORE_CLASSES[ci],
                             "bbox": [bx1 + x0, by1 + y0, bx2 + x0, by2 + y0],
                             "score": float(b.conf)})
        det_only = nms_per_label(dets, args.nms_iou)
        n_det += len(det_only)

        # 与 VLM 伪标注融合(并集后再 NMS 去重)
        vlm = vlm_by_file.get(fn, [])
        n_vlm += len(vlm)
        fused = nms_per_label(det_only + vlm, args.nms_iou)

        for d in fused:
            bx1, by1, bx2, by2 = d["bbox"]
            w, h = bx2 - bx1, by2 - by1
            if w <= 0 or h <= 0:
                continue
            anns.append({"id": aid, "image_id": iid, "category_id": cat_id[d["label"]],
                         "bbox": [bx1, by1, w, h], "area": w * h, "iscrowd": 0,
                         "score": d["score"]})
            aid += 1

    cats = [{"id": i + 1, "name": c} for i, c in enumerate(CORE_CLASSES)]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": anns, "categories": cats}, f)
    print(f"[sahi_relabel] images={len(images)}  det_tiled_boxes={n_det}  "
          f"+vlm={n_vlm}  -> fused_labels={len(anns)}  -> {args.out}")


if __name__ == "__main__":
    main()
