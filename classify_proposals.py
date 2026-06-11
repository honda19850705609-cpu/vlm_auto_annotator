#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classify_proposals.py —— 解耦范式第二步:给 SAM 候选框分类(攻精度)

SAM 出的是类无关候选框(高召回、低精度,含背景)。本步用 CLIP 零样本给每个候选
crop 分类成 pedestrian/vehicle/bicycle 或 background,保留核心三类、丢背景,得到
干净的标注。CLIP 是标准的 SAM 配套分类器(SAM+CLIP),批量快。

关键:每个框带**上下文边距**裁剪(context×)再分类——单看 8px 的框信息太少,带点
周边路面/环境,CLIP 更容易判对。

依赖(5090):pip install open_clip_torch
  权重首次自动下载(加速开了即可)。

输入:SAM 候选 COCO(sam_propose.py 产物)+ 原图。
输出:带核心类标签的 COCO(可直接 badcase.py 对照 GT)。

用法:
  python classify_proposals.py --prop sam_prop.json --images smoke5/ \
      --out sam_clip_coco.json --context 2.0 --margin 0.0
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image
from badcase import CORE_CLASSES   # pedestrian / vehicle / bicycle

# 每个核心类的文本提示(多模板平均更稳);最后一组是背景(判到则丢弃)
CLASS_PROMPTS = {
    "pedestrian": ["an aerial top-down photo of a pedestrian", "a person walking seen from above"],
    "vehicle":    ["an aerial top-down photo of a car", "a car / van / truck / bus seen from above"],
    "bicycle":    ["an aerial top-down photo of a bicycle or motorcycle", "a two-wheeler seen from above"],
    "__bg__":     ["an aerial photo of road / ground / rooftop / tree / background", "empty pavement seen from above"],
}


def main():
    ap = argparse.ArgumentParser(description="CLIP zero-shot classify SAM proposals -> labeled COCO")
    ap.add_argument("--prop", required=True, help="SAM 候选 COCO")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", type=float, default=2.0, help="裁剪边距倍数(框的 N 倍区域)")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="核心类最高分需比背景高出此 margin 才保留(0=只要 argmax 是核心类)")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    import torch
    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    # 预编码文本(每类多模板取平均)
    names = list(CLASS_PROMPTS.keys())
    with torch.no_grad():
        text_feats = []
        for nm in names:
            toks = tokenizer(CLASS_PROMPTS[nm]).to(device)
            tf = model.encode_text(toks)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            text_feats.append(tf.mean(0))
        text_feats = torch.stack(text_feats)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    coco = json.load(open(args.prop))
    imgid2file = {im["id"]: im["file_name"] for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    cat_id = {c: i + 1 for i, c in enumerate(CORE_CLASSES)}
    out_images = list(coco["images"])
    out_anns = []
    aid = 1
    kept = dropped_bg = 0

    for im in coco["images"]:
        fn = imgid2file[im["id"]]
        path = os.path.join(args.images, fn)
        if not os.path.exists(path):
            continue
        pil = Image.open(path).convert("RGB")
        W, H = pil.size
        props = anns_by_img.get(im["id"], [])

        # 批量裁剪+预处理
        crops, meta = [], []
        for a in props:
            x, y, w, h = a["bbox"]
            cx, cy = x + w / 2, y + h / 2
            half_w, half_h = w * args.context / 2, h * args.context / 2
            l, t = max(0, int(cx - half_w)), max(0, int(cy - half_h))
            r, b = min(W, int(cx + half_w)), min(H, int(cy + half_h))
            if r - l < 2 or b - t < 2:
                continue
            crops.append(preprocess(pil.crop((l, t, r, b))))
            meta.append(a)

        # 分批 CLIP 前向
        for i in range(0, len(crops), args.batch):
            batch = torch.stack(crops[i:i + args.batch]).to(device)
            with torch.no_grad():
                feat = model.encode_image(batch)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                logits = (100.0 * feat @ text_feats.T)
                probs = logits.softmax(dim=-1).cpu().numpy()
            for j, a in enumerate(meta[i:i + args.batch]):
                p = probs[j]
                k = int(p.argmax())
                cls = names[k]
                if cls == "__bg__":
                    dropped_bg += 1
                    continue
                core_best = max(p[:3])
                if core_best - p[3] < args.margin:   # 和背景差距不够 -> 丢
                    dropped_bg += 1
                    continue
                x, y, w, h = a["bbox"]
                out_anns.append({"id": aid, "image_id": im["id"], "category_id": cat_id[cls],
                                 "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
                                 "score": float(core_best)})
                aid += 1
                kept += 1
        print(f"  {fn}: {len(props)} props -> kept {kept-0 if False else ''}", end="\r")

    cats = [{"id": i + 1, "name": c} for i, c in enumerate(CORE_CLASSES)]
    json.dump({"images": out_images, "annotations": out_anns, "categories": cats},
              open(args.out, "w"))
    print(f"\n[OK] kept {kept} (dropped bg/low-conf {dropped_bg}) -> {args.out}")


if __name__ == "__main__":
    main()
