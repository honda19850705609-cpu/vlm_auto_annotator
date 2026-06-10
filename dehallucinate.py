#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dehallucinate.py  ——  去"等距网格幻觉"后处理(精确率杠杆)

置信度分析证明 VLM 自报分数无法筛 FP(分数全挤高位)。FP 的主要来源之一是
**网格幻觉**:模型被要求数密集小目标时,会吐出一长串"同标签 + 尺寸像素级相同 +
沿一条直线等距排列"的假框(真实目标有自然抖动,不会像素级整齐)。

本脚本对一个 COCO 检测 json 做后处理:按 (image, category) 分组,找
**尺寸近乎相同 (w,h 取整后一致) 且中心共线 (一轴方差极小)** 的成簇框,簇长
>= min_run 即判为幻觉、整簇丢弃。判据故意保守(要求像素级同尺寸 + 近乎零抖动),
以放过真实成排停车(后者尺寸/位置有自然变化)。

纯 CPU,输入输出都是 COCO json,可直接接 badcase.py 评估增益。

用法:
  python dehallucinate.py --in vlm_coco.json --out vlm_coco_dehall.json \
      --min-run 6 --size-tol 1.5 --line-tol 4.0
"""

import argparse
import json
from collections import defaultdict
from statistics import pstdev


def find_hallucinated(anns, min_run, size_tol, line_tol):
    """返回该 (image,category) 分组里判为幻觉的 ann 索引集合。

    anns: list of coco annotation dict(同一图同一类)。
    判据:按尺寸分桶(round 到整数 px,容差 size_tol),桶内 >=min_run 个框,
    且中心点在某一轴上的标准差 < line_tol(共线)→ 整桶判幻觉。
    """
    flagged = set()
    # 尺寸桶:用整数化的 (w,h) 作 key,容差靠四舍五入吸收
    buckets = defaultdict(list)
    for i, a in enumerate(anns):
        _, _, w, h = a["bbox"]
        key = (round(w / size_tol), round(h / size_tol))
        buckets[key].append(i)

    for key, idxs in buckets.items():
        if len(idxs) < min_run:
            continue
        cxs = [anns[i]["bbox"][0] + anns[i]["bbox"][2] / 2 for i in idxs]
        cys = [anns[i]["bbox"][1] + anns[i]["bbox"][3] / 2 for i in idxs]
        # 共线:某一轴几乎不变(像素级整齐的行/列)
        if pstdev(cxs) < line_tol or pstdev(cys) < line_tol:
            flagged.update(idxs)
    return flagged


def dehallucinate(coco, min_run, size_tol, line_tol):
    by_img_cat = defaultdict(list)
    for a in coco["annotations"]:
        by_img_cat[(a["image_id"], a["category_id"])].append(a)

    drop_ids = set()
    grids = 0
    for (_imgid, _cat), anns in by_img_cat.items():
        flagged = find_hallucinated(anns, min_run, size_tol, line_tol)
        if flagged:
            grids += 1
            for i in flagged:
                drop_ids.add(id(anns[i]))

    kept = [a for a in coco["annotations"] if id(a) not in drop_ids]
    dropped = len(coco["annotations"]) - len(kept)
    out = {k: v for k, v in coco.items() if k != "annotations"}
    out["annotations"] = kept
    return out, dropped, grids


def main():
    ap = argparse.ArgumentParser(description="Drop evenly-spaced grid hallucinations from a COCO det json")
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-run", type=int, default=6,
                    help="同尺寸共线成簇 >= 此数才判幻觉 (默认 6)")
    ap.add_argument("--size-tol", type=float, default=1.5,
                    help="尺寸分桶容差(px);越小越严 (默认 1.5)")
    ap.add_argument("--line-tol", type=float, default=4.0,
                    help="共线判定:中心某轴 std < 此值(px) (默认 4.0)")
    args = ap.parse_args()

    with open(args.in_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    n0 = len(coco["annotations"])
    out, dropped, grids = dehallucinate(coco, args.min_run, args.size_tol, args.line_tol)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[dehall] 输入框={n0}  丢弃幻觉框={dropped}  涉及网格簇={grids}  "
          f"剩余={n0 - dropped}")
    print(f"[OK] -> {args.out}")


if __name__ == "__main__":
    main()
