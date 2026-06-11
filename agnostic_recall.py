#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agnostic_recall.py —— 类无关召回(只看"定位",不看类别)

回答"定位器(如 SAM)到底找没找到这些目标",不管它叫什么。每个 GT 框只要有
**任意一个**候选框 IoU>=阈值 就算命中。按目标尺寸分档,看小目标到底有没有被定位。

用于裁决"解耦定位+分类"范式:若 SAM 的类无关召回(尤其 <8/8-16px)远高于 VLM
自己定位的召回,说明定位器能补 VLM 的短板,范式成立;若也低,范式不通。

用法:
  python agnostic_recall.py --gt gt5.json --prop sam_prop.json
"""

import argparse
import json
from collections import defaultdict

from badcase import iou_xywh, size_bin_of, SIZE_BINS, DEFAULT_LABEL_MAP


def load_boxes(path, core_only):
    """返回 {file_name: [xywh,...]}。core_only=True 时只留核心类(给 GT 用)。"""
    data = json.load(open(path))
    id2name = {c["id"]: c["name"] for c in data.get("categories", [])}
    imgid2file = {im["id"]: im["file_name"] for im in data.get("images", [])}
    out = defaultdict(list)
    for a in data.get("annotations", []):
        fn = imgid2file.get(a["image_id"])
        if fn is None:
            continue
        if core_only:
            cls = DEFAULT_LABEL_MAP.get(str(id2name.get(a["category_id"], "")).strip().lower())
            if cls is None:
                continue
        out[fn].append(a["bbox"])
    return out


def main():
    ap = argparse.ArgumentParser(description="Class-agnostic localization recall (per size)")
    ap.add_argument("--gt", required=True, help="真值 COCO(核心类)")
    ap.add_argument("--prop", required=True, help="候选框 COCO(类无关)")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    gt = load_boxes(args.gt, core_only=True)
    prop = load_boxes(args.prop, core_only=False)

    bin_tp = defaultdict(int)
    bin_tot = defaultdict(int)
    tp = tot = 0
    for fn, gboxes in gt.items():
        pboxes = prop.get(fn, [])
        for gb in gboxes:
            b = size_bin_of(gb[2] * gb[3])
            bin_tot[b] += 1
            tot += 1
            if any(iou_xywh(gb, pb) >= args.iou for pb in pboxes):
                bin_tp[b] += 1
                tp += 1

    print(f"# 类无关定位召回 (IoU={args.iou})")
    print(f"- 总体: recall={tp/tot if tot else 0:.3f}  ({tp}/{tot})")
    print(f"- 候选框总数: {sum(len(v) for v in prop.values())}")
    for name, _, _ in SIZE_BINS:
        t, n = bin_tp.get(name, 0), bin_tot.get(name, 0)
        print(f"    - {name:<6} recall={t/n if n else 0:.3f}  ({t}/{n})")


if __name__ == "__main__":
    main()
