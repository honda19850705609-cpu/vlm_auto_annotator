#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
to_coco.py  ——  D6 端到端链路收口

把 structured_vlm.py 批量模式产出的 annotations.json
转成标准 COCO detection 格式,并用 pycocotools 验证能被加载。

链路:  图像文件夹 -> structured_vlm(批量) -> annotations.json
       -> [本脚本] -> coco.json -> pycocotools.COCO() load 通过

输入 JSON 结构(实测):
{
  "model": ..., "num_images": N, ...,
  "annotations": {
    "img1.jpg": [{"label": "car", "bbox": [x1,y1,x2,y2], "confidence": 0.9}, ...],
    ...
  }
}
bbox 为原图像素坐标 [x1, y1, x2, y2](已 rescale)。

用法:
  python to_coco.py --in annotations.json --images images/ --out coco.json
  python to_coco.py --in annotations.json --images images/ --out coco.json --verify
"""

import argparse
import json
import os
import sys
from PIL import Image


def build_category_map(annotations):
    """扫描所有 label,排序后分配稳定的 category_id(从 1 开始,COCO 惯例)。

    排序是关键:只要 label 集合相同,id 映射就可复现,
    不受 label 在数据里出现顺序的影响。
    """
    labels = set()
    for dets in annotations.values():
        for d in dets:
            labels.add(d["label"])
    sorted_labels = sorted(labels)  # 字母序固定
    # COCO category_id 从 1 开始(0 常被保留)
    cat_map = {name: i + 1 for i, name in enumerate(sorted_labels)}
    categories = [
        {"id": cid, "name": name, "supercategory": "object"}
        for name, cid in cat_map.items()
    ]
    return cat_map, categories


def xyxy_to_xywh(bbox):
    """[x1,y1,x2,y2] -> COCO [x,y,w,h];顺带做防御性 clamp(w/h 不为负)。"""
    x1, y1, x2, y2 = bbox
    # 防御:个别 VLM 输出可能左右/上下颠倒
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    w = x_max - x_min
    h = y_max - y_min
    return [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)], w * h


def convert(in_path, images_dir, out_path):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations", {})
    if not annotations:
        sys.exit("[ERR] 输入 JSON 里没有 annotations 段,检查文件。")

    cat_map, categories = build_category_map(annotations)

    images = []
    coco_anns = []
    ann_id = 1
    missing_imgs = []

    # 文件名排序,保证 image_id 也稳定可复现
    for image_id, fname in enumerate(sorted(annotations.keys()), start=1):
        img_path = os.path.join(images_dir, fname)
        if not os.path.exists(img_path):
            missing_imgs.append(fname)
            # 找不到图就跳过该图的宽高(但仍记录,避免下游图数对不上)
            width = height = 0
        else:
            with Image.open(img_path) as im:
                width, height = im.size

        # 空检测的图也要进 images 段(否则评估时图数对不上)
        images.append({
            "id": image_id,
            "file_name": fname,
            "width": width,
            "height": height,
        })

        for d in annotations[fname]:
            bbox_xywh, area = xyxy_to_xywh(d["bbox"])
            coco_anns.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_map[d["label"]],
                "bbox": bbox_xywh,
                "area": round(area, 2),
                "iscrowd": 0,
                "score": d.get("confidence", 1.0),  # 保留置信度,评估时可用
            })
            ann_id += 1

    coco = {
        "info": {"description": "VLM auto-annotation -> COCO", "source": in_path},
        "images": images,
        "annotations": coco_anns,
        "categories": categories,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"[OK] 写出 {out_path}")
    print(f"     images={len(images)}  annotations={len(coco_anns)}  categories={len(categories)}")
    print(f"     类别映射: {cat_map}")
    if missing_imgs:
        print(f"[WARN] {len(missing_imgs)} 张图在 {images_dir} 找不到,宽高记为 0: {missing_imgs}")
    return out_path


def verify(coco_path):
    """用 pycocotools 实例化并跑通基本接口,作为端到端硬验证。"""
    try:
        from pycocotools.coco import COCO
    except ImportError:
        sys.exit("[ERR] 未装 pycocotools,跑: pip install pycocotools")

    # COCO() 会打印 loading 信息,正常
    coco = COCO(coco_path)
    cat_ids = coco.getCatIds()
    img_ids = coco.getImgIds()
    ann_ids = coco.getAnnIds()
    cats = coco.loadCats(cat_ids)
    print("\n[VERIFY] pycocotools 加载成功 ✓")
    print(f"  categories ({len(cat_ids)}): {[c['name'] for c in cats]}")
    print(f"  images: {len(img_ids)}   annotations: {len(ann_ids)}")
    # 抽一张图看它的标注能正常取出
    if img_ids:
        sample_anns = coco.loadAnns(coco.getAnnIds(imgIds=img_ids[0]))
        print(f"  样例 image_id={img_ids[0]} 取出 {len(sample_anns)} 条标注 ✓")
    print("[VERIFY] 端到端链路闭环通过。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="annotations.json 路径")
    ap.add_argument("--images", required=True, help="images/ 文件夹路径")
    ap.add_argument("--out", default="coco.json", help="输出 COCO json 路径")
    ap.add_argument("--verify", action="store_true", help="转换后用 pycocotools 验证")
    args = ap.parse_args()

    out = convert(args.in_path, args.images, args.out)
    if args.verify:
        verify(out)
