#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sam_propose.py —— SAM(Segment Anything)类无关候选框生成器

新范式第一步:用 SAM 的 everything 模式把图里**所有独立物体**分割出来(类无关、
高召回,含极小目标),把每个 mask 转成 bbox,输出成 COCO(单一 "object" 类)。

这一步**不含 VLM**——目的是先单独验证:SAM 能不能把 VisDrone 的小目标定位出来。
用 agnostic_recall.py 量它的类无关召回(尤其 <8/8-16px)。若 SAM 召回高,再接
VLM 分类那一步;若 SAM 也漏小目标,这条范式同样不通,早止损。

依赖(5090,加速已开):
  pip install segment-anything opencv-python
  # 权重(三选一,vit_b 最小最快 ~375MB):
  #   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

切图跑 SAM:整图直接 SAM 对小目标不利(下采样),所以也支持 --tile 把图切块后
逐块 SAM,框拼回全图——和我们切图标注同理,给小目标更高有效分辨率。

用法:
  python sam_propose.py --images smoke5/ --ckpt sam_vit_b_01ec64.pth --model-type vit_b \
      --out sam_prop.json --tile 640 --overlap 0.2 --points-per-side 48
"""

import argparse
import json
import os

import numpy as np
from PIL import Image
from yolo_to_coco import _make_tiles, _nms_per_class


def main():
    ap = argparse.ArgumentParser(description="SAM everything-mode class-agnostic box proposals")
    ap.add_argument("--images", required=True)
    ap.add_argument("--ckpt", required=True, help="SAM 权重 .pth")
    ap.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    ap.add_argument("--out", required=True, help="输出 COCO(单类 object)")
    ap.add_argument("--tile", type=int, default=0, help="切块边长;0=整图(密集小目标建议 640)")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--points-per-side", type=int, default=32,
                    help="SAM 采样网格密度;小目标多建议调高(48-64,更慢)")
    ap.add_argument("--min-area", type=float, default=4.0, help="丢弃面积过小的 mask(像素)")
    ap.add_argument("--max-area-frac", type=float, default=0.2,
                    help="丢弃面积 > 块面积*此值 的 mask(背景大块)")
    args = ap.parse_args()

    import torch
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry[args.model_type](checkpoint=args.ckpt)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    gen = SamAutomaticMaskGenerator(
        sam, points_per_side=args.points_per_side,
        pred_iou_thresh=0.80, stability_score_thresh=0.85, min_mask_region_area=4)

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
    files = sorted(f for f in os.listdir(args.images) if f.lower().endswith(IMG_EXTS))
    images, anns = [], []
    aid = 1
    for iid, fn in enumerate(files, 1):
        path = os.path.join(args.images, fn)
        pil = Image.open(path).convert("RGB")
        W, H = pil.size
        images.append({"id": iid, "file_name": fn, "width": W, "height": H})
        arr = np.array(pil)

        tiles = _make_tiles(W, H, args.tile, args.overlap) if args.tile > 0 else [(0, 0, W, H)]
        dets = []  # (cls=0, x1,y1,x2,y2, score)
        for (x0, y0, x1, y1) in tiles:
            crop = arr[y0:y1, x0:x1]
            tile_area = (x1 - x0) * (y1 - y0)
            for m in gen.generate(crop):
                bx, by, bw, bh = m["bbox"]   # 块内 xywh
                area = bw * bh
                if area < args.min_area or area > args.max_area_frac * tile_area:
                    continue
                dets.append((0, bx + x0, by + y0, bx + x0 + bw, by + y0 + bh,
                             float(m.get("predicted_iou", 1.0))))
        merged = _nms_per_class(dets, 0.7)
        for _, X1, Y1, X2, Y2, sc in merged:
            anns.append({"id": aid, "image_id": iid, "category_id": 1,
                         "bbox": [X1, Y1, X2 - X1, Y2 - Y1], "area": (X2 - X1) * (Y2 - Y1),
                         "iscrowd": 0, "score": sc})
            aid += 1
        print(f"  [{iid}/{len(files)}] {fn}: {len(merged)} proposals")

    coco = {"images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "object"}]}
    with open(args.out, "w") as f:
        json.dump(coco, f)
    print(f"[OK] {len(anns)} proposals -> {args.out}")


if __name__ == "__main__":
    main()
