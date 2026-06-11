#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classify_proposals_vlm.py —— 解耦范式第二步(强版):Qwen-VL 给 SAM 候选框分类

CLIP 在极小航拍 crop 上分不清目标/背景(精度崩到 0.09)。换 Qwen2.5-VL 当分类器:
它能真正推理"这块是车 / 行人 / 自行车 / 还是背景"。

效率:6496 个 crop 不能逐个调用。用 **montage 批量**——把 grid×grid 个 crop(各带
上下文、缩到统一尺寸)拼成一张网格图,一次 Qwen 调用按阅读顺序输出 grid² 个标签。
~25/次 → 5 张图约 260 次调用。

输入:SAM 候选 COCO + 原图。输出:带核心类标签的 COCO(丢 background)。

用法:
  python classify_proposals_vlm.py --prop sam_prop.json --images smoke5/ \
      --model /root/autodl-tmp/qwen2.5-vl-7b --out sam_qwen_coco.json \
      --grid 5 --cell 112 --context 2.5
"""

import argparse
import json
import os
from collections import defaultdict

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

from structured_vlm import load_model, extract_json
from badcase import CORE_CLASSES

ALLOWED = {"pedestrian": "pedestrian", "people": "pedestrian", "person": "pedestrian",
           "vehicle": "vehicle", "car": "vehicle", "van": "vehicle", "truck": "vehicle",
           "bus": "vehicle", "bicycle": "bicycle", "bike": "bicycle", "motor": "bicycle",
           "motorcycle": "bicycle", "tricycle": "bicycle"}


def crop_with_context(pil, box, context):
    W, H = pil.size
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    hw, hh = w * context / 2, h * context / 2
    l, t = max(0, int(cx - hw)), max(0, int(cy - hh))
    r, b = min(W, int(cx + hw)), min(H, int(cy + hh))
    if r - l < 2 or b - t < 2:
        return None
    return pil.crop((l, t, r, b))


def build_montage(crops, grid, cell):
    """把 <=grid² 个 crop 拼成 grid×grid 网格图(缺的留黑)。"""
    canvas = Image.new("RGB", (grid * cell, grid * cell), (0, 0, 0))
    for i, c in enumerate(crops):
        if i >= grid * grid:
            break
        r, col = divmod(i, grid)
        canvas.paste(c.resize((cell, cell)), (col * cell, r * cell))
    return canvas


def classify_montage(model, processor, montage, n, grid, max_new_tokens):
    prompt = (
        f"This is a {grid}x{grid} grid of {n} small image crops in reading order "
        f"(left to right, top to bottom). Each crop is centered on one candidate object "
        f"from an aerial/drone view. For EACH of the first {n} crops, classify it as "
        f"exactly one of: pedestrian, vehicle, bicycle, none "
        f"(use 'none' if it is background / not one of these). "
        f"Respond with ONLY a JSON array of {n} strings, one per crop in reading order."
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": montage}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(messages)
    inputs = processor(text=[text], images=img_in, videos=vid_in,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    try:
        labels = extract_json(raw)
    except ValueError:
        return ["none"] * n
    out = []
    for v in labels[:n]:
        out.append(str(v).strip().lower() if isinstance(v, str) else "none")
    out += ["none"] * (n - len(out))
    return out


def main():
    ap = argparse.ArgumentParser(description="Qwen-VL classify SAM proposals via montage batching")
    ap.add_argument("--prop", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", type=int, default=5, help="每张 montage 边长(grid×grid 个 crop)")
    ap.add_argument("--cell", type=int, default=112, help="每个 crop 在 montage 里的边长 px")
    ap.add_argument("--context", type=float, default=2.5, help="裁剪上下文倍数")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    model, processor = load_model(args.model)
    coco = json.load(open(args.prop))
    imgid2file = {im["id"]: im["file_name"] for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    cat_id = {c: i + 1 for i, c in enumerate(CORE_CLASSES)}
    per = args.grid * args.grid
    out_anns = []
    aid = 1
    kept = bg = 0

    for im in coco["images"]:
        fn = imgid2file[im["id"]]
        path = os.path.join(args.images, fn)
        if not os.path.exists(path):
            continue
        pil = Image.open(path).convert("RGB")
        props, crops = [], []
        for a in anns_by_img.get(im["id"], []):
            c = crop_with_context(pil, a["bbox"], args.context)
            if c is not None:
                props.append(a)
                crops.append(c)

        for i in range(0, len(crops), per):
            chunk_crops = crops[i:i + per]
            chunk_props = props[i:i + per]
            n = len(chunk_crops)
            montage = build_montage(chunk_crops, args.grid, args.cell)
            labels = classify_montage(model, processor, montage, n, args.grid, args.max_new_tokens)
            for a, lab in zip(chunk_props, labels):
                cls = ALLOWED.get(lab)
                if cls is None:
                    bg += 1
                    continue
                x, y, w, h = a["bbox"]
                out_anns.append({"id": aid, "image_id": im["id"], "category_id": cat_id[cls],
                                 "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
                                 "score": float(a.get("score", 1.0))})
                aid += 1
                kept += 1
        print(f"  {fn}: {len(props)} props -> kept {kept}, bg {bg}")

    cats = [{"id": i + 1, "name": c} for i, c in enumerate(CORE_CLASSES)]
    json.dump({"images": coco["images"], "annotations": out_anns, "categories": cats},
              open(args.out, "w"))
    print(f"[OK] kept {kept} (dropped background {bg}) -> {args.out}")


if __name__ == "__main__":
    main()
