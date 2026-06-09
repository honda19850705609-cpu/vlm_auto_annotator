#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
badcase.py  ——  D9 + D11  GT 锚定的 badcase 分析 / 量化

用 VisDrone 人工真值(GT)当锚,量化:
  (1) 两方:VLM 伪标注 vs GT  —— 回答「VLM 自动标注质量到底多差」
      产出: 总体 P/R/F1 + 按目标尺寸分档的召回率(小目标是 VLM 的命门)
  (2) 三方(可选,给了 --pred 才算):小模型(DINO-DETR)预测 vs GT
      额外产出杀手指标: 「在小模型漏检(FN)的真值上,VLM 能补救多少」
                       = VLM 在 pred-FN 子集上的召回率 (rescue rate)

三套标注都吃标准 COCO instance json(images/annotations/categories)。
本脚本纯 CPU、只读 json,不依赖 GPU、不依赖 pycocotools。

关键工程点:
  * 按 file_name 对齐,而不是 image_id —— 三套 json 的 id 编号规则不同。
  * 标签归一化 —— GT 是合并类 (pedestrian/vehicle/bicycle),
    VLM 吐的是自由词 (car/truck/person...),必须先映射到同一套核心类才能比。
  * 匹配用每图每类的贪心 IoU(降序),IoU>=阈值算命中。透明、可解释,
    不走 COCOeval,方便你在报告里讲清楚每个 badcase 怎么来的。

用法:
  # 两方(零依赖小模型)
  python badcase.py --gt gt_coco.json --vlm vlm_coco.json --out badcase_out/
  # 三方(加上小模型预测)
  python badcase.py --gt gt_coco.json --vlm vlm_coco.json --pred dino_coco.json --out badcase_out/
"""

import argparse
import json
import os
from collections import defaultdict


# ----------------------------------------------------------------------------
# 标签归一化:把任意标注的类别名映射到 VisDrone 核心三类。
# 与 data_processing.py 的 CORE_GROUPS 对齐:
#   pedestrian = {pedestrian, people}
#   vehicle    = {car, van, truck, bus}
#   bicycle    = {bicycle}  (motor/tricycle 这里也并进非机动,可按需调)
# 映射不到的类别 -> None(默认丢弃,不参与对比)。
# 全部小写后匹配;VLM 自由词尽量覆盖常见同义词。
# ----------------------------------------------------------------------------
DEFAULT_LABEL_MAP = {
    # pedestrian
    "pedestrian": "pedestrian", "people": "pedestrian", "person": "pedestrian",
    "human": "pedestrian", "man": "pedestrian", "woman": "pedestrian",
    # vehicle
    "vehicle": "vehicle", "car": "vehicle", "van": "vehicle", "truck": "vehicle",
    "bus": "vehicle", "suv": "vehicle", "minivan": "vehicle", "pickup": "vehicle",
    # bicycle / 两轮
    "bicycle": "bicycle", "bike": "bicycle", "cycle": "bicycle",
    "motor": "bicycle", "motorcycle": "bicycle", "motorbike": "bicycle",
    "scooter": "bicycle", "tricycle": "bicycle",
}

CORE_CLASSES = ["pedestrian", "vehicle", "bicycle"]

# 目标尺寸分档(按 sqrt(area) 的像素尺度);小目标是 aerial 场景的核心难点。
SIZE_BINS = [
    ("<8",      0,    8),
    ("8-16",    8,    16),
    ("16-32",   16,   32),
    ("32-96",   32,   96),
    (">=96",    96,   1e9),
]


def size_bin_of(area):
    """按 sqrt(area) 落到尺寸档。area 为像素面积。"""
    scale = area ** 0.5
    for name, lo, hi in SIZE_BINS:
        if lo <= scale < hi:
            return name
    return SIZE_BINS[-1][0]


def normalize_label(name, label_map):
    return label_map.get(str(name).strip().lower())


def load_coco_by_filename(path, label_map):
    """
    读 COCO instance json,返回:
      dets_by_file: {file_name: [ {cls, bbox[xywh], area, score} ]}
      size_by_file: {file_name: (w, h)}  # 图像尺寸,缺失为 (0,0)
    类别经过归一化,映射不到核心类的标注直接丢弃。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id2name = {c["id"]: c["name"] for c in data.get("categories", [])}
    imgid2file = {im["id"]: im["file_name"] for im in data.get("images", [])}
    size_by_file = {
        im["file_name"]: (im.get("width", 0), im.get("height", 0))
        for im in data.get("images", [])
    }

    dets_by_file = defaultdict(list)
    dropped = 0
    for a in data.get("annotations", []):
        fname = imgid2file.get(a["image_id"])
        if fname is None:
            continue
        raw_name = id2name.get(a["category_id"], str(a["category_id"]))
        cls = normalize_label(raw_name, label_map)
        if cls is None:
            dropped += 1
            continue
        x, y, w, h = a["bbox"]
        dets_by_file[fname].append({
            "cls": cls,
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": float(a.get("area", w * h)),
            "score": float(a.get("score", 1.0)),
        })
    return dets_by_file, size_by_file, dropped


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_one_image(gt_dets, pred_dets, iou_thr):
    """
    每图每类贪心匹配(pred 按 score 降序逐个找未占用、IoU 最大且>=阈值的 gt)。
    返回:
      tp_gt_idx: 命中的 gt 下标集合
      matched: list of (pred_idx, gt_idx, iou)
      fp_pred_idx: 误检的 pred 下标集合
    gt_dets / pred_dets: 同一张图的检测列表。
    """
    matched = []
    used_gt = set()
    # 按类分组 gt
    gt_by_cls = defaultdict(list)
    for gi, g in enumerate(gt_dets):
        gt_by_cls[g["cls"]].append(gi)

    fp_pred_idx = set()
    order = sorted(range(len(pred_dets)), key=lambda i: -pred_dets[i]["score"])
    for pi in order:
        p = pred_dets[pi]
        best_iou, best_gi = 0.0, -1
        for gi in gt_by_cls.get(p["cls"], []):
            if gi in used_gt:
                continue
            iou = iou_xywh(p["bbox"], gt_dets[gi]["bbox"])
            if iou >= iou_thr and iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi >= 0:
            used_gt.add(best_gi)
            matched.append((pi, best_gi, best_iou))
        else:
            fp_pred_idx.add(pi)
    tp_gt_idx = set(g for _, g, _ in matched)
    return tp_gt_idx, matched, fp_pred_idx


def evaluate(gt_by_file, src_by_file, iou_thr):
    """
    把 src(VLM 或 pred)对照 GT 评估。返回逐图结果 + 汇总计数器。
    per_file[fname] = {tp, fp, fn, fn_gt_idx:[...], gt:[...]}  方便后续补救率 & badcase 排序。
    """
    per_file = {}
    # 汇总
    glob = {"tp": 0, "fp": 0, "fn": 0}
    by_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    # 召回按 gt 尺寸档
    bin_tp = defaultdict(int)
    bin_total = defaultdict(int)

    all_files = set(gt_by_file) | set(src_by_file)
    for fname in all_files:
        gt = gt_by_file.get(fname, [])
        src = src_by_file.get(fname, [])
        tp_gt_idx, matched, fp_pred_idx = match_one_image(gt, src, iou_thr)

        fn_gt_idx = [gi for gi in range(len(gt)) if gi not in tp_gt_idx]
        tp, fp, fn = len(tp_gt_idx), len(fp_pred_idx), len(fn_gt_idx)
        glob["tp"] += tp; glob["fp"] += fp; glob["fn"] += fn

        for gi in tp_gt_idx:
            by_class[gt[gi]["cls"]]["tp"] += 1
            bin_tp[size_bin_of(gt[gi]["area"])] += 1
        for gi in fn_gt_idx:
            by_class[gt[gi]["cls"]]["fn"] += 1
        for pi in fp_pred_idx:
            by_class[src[pi]["cls"]]["fp"] += 1
        for gi in range(len(gt)):
            bin_total[size_bin_of(gt[gi]["area"])] += 1

        per_file[fname] = {
            "tp": tp, "fp": fp, "fn": fn,
            "fn_gt_idx": fn_gt_idx,
            "tp_gt_idx": sorted(tp_gt_idx),
        }
    return {
        "per_file": per_file,
        "global": glob,
        "by_class": dict(by_class),
        "bin_tp": dict(bin_tp),
        "bin_total": dict(bin_total),
    }


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def fmt_block(title, ev):
    g = ev["global"]
    p, r, f = prf(g["tp"], g["fp"], g["fn"])
    lines = [f"### {title} vs GT",
             f"- 总体: P={p:.3f}  R={r:.3f}  F1={f:.3f}  (TP={g['tp']} FP={g['fp']} FN={g['fn']})",
             "- 分类别:"]
    for cls in CORE_CLASSES:
        c = ev["by_class"].get(cls, {"tp": 0, "fp": 0, "fn": 0})
        cp, cr, cf = prf(c["tp"], c["fp"], c["fn"])
        lines.append(f"    - {cls:<11} P={cp:.3f} R={cr:.3f} F1={cf:.3f} "
                     f"(TP={c['tp']} FP={c['fp']} FN={c['fn']})")
    lines.append("- 按目标尺寸分档的召回率(小目标是命门):")
    for name, _, _ in SIZE_BINS:
        tot = ev["bin_total"].get(name, 0)
        tp = ev["bin_tp"].get(name, 0)
        rec = tp / tot if tot else 0.0
        lines.append(f"    - {name:<6} recall={rec:.3f}  ({tp}/{tot})")
    return "\n".join(lines)


def badcase_ranking(gt_by_file, vlm_ev, top_n):
    """
    badcase 排序:每图 score = 漏检真值数,且小目标加权(越小权重越高)。
    weight(area) = clamp(32 / sqrt(area), 1, 8)  —— 32px 以下越小越值钱。
    """
    rows = []
    for fname, fr in vlm_ev["per_file"].items():
        gt = gt_by_file.get(fname, [])
        weighted = 0.0
        small_miss = 0
        for gi in fr["fn_gt_idx"]:
            scale = max(1.0, gt[gi]["area"] ** 0.5)
            w = min(8.0, max(1.0, 32.0 / scale))
            weighted += w
            if scale < 32:
                small_miss += 1
        rows.append({
            "file_name": fname,
            "fn": fr["fn"], "fp": fr["fp"], "tp": fr["tp"],
            "small_miss": small_miss,
            "badcase_score": round(weighted, 2),
        })
    rows.sort(key=lambda x: -x["badcase_score"])
    return rows[:top_n] if top_n > 0 else rows


def rescue_metric(gt_by_file, pred_ev, vlm_by_file, iou_thr):
    """
    杀手指标: 小模型漏检(pred FN)的真值里,VLM 能补回多少。
    对每张图,取 pred 的 fn_gt_idx,看这些 gt 是否被 VLM 命中(IoU>=阈值,同类)。
    rescue_rate = VLM 命中的 pred-FN 数 / pred-FN 总数
    """
    rescued, total = 0, 0
    per_file = {}
    for fname, fr in pred_ev["per_file"].items():
        gt = gt_by_file.get(fname, [])
        vlm = vlm_by_file.get(fname, [])
        fn_idx = fr["fn_gt_idx"]
        if not fn_idx:
            continue
        hit = 0
        for gi in fn_idx:
            g = gt[gi]
            ok = any(
                v["cls"] == g["cls"] and iou_xywh(v["bbox"], g["bbox"]) >= iou_thr
                for v in vlm
            )
            if ok:
                hit += 1
        rescued += hit
        total += len(fn_idx)
        per_file[fname] = {"pred_fn": len(fn_idx), "vlm_rescued": hit}
    rate = rescued / total if total else 0.0
    return {"rescued": rescued, "pred_fn_total": total, "rescue_rate": rate,
            "per_file": per_file}


def main():
    ap = argparse.ArgumentParser(description="GT 锚定的 VLM/小模型 badcase 分析 (D9+D11)")
    ap.add_argument("--gt", required=True, help="VisDrone 真值 COCO json")
    ap.add_argument("--vlm", required=True, help="VLM 伪标注 COCO json (to_coco.py 产物)")
    ap.add_argument("--pred", default=None, help="小模型(DINO-DETR)预测 COCO json (可选,给了就做三方+补救率)")
    ap.add_argument("--iou", type=float, default=0.5, help="匹配 IoU 阈值 (默认 0.5)")
    ap.add_argument("--top-n", type=int, default=50, help="badcase 排行输出前 N 张 (默认 50)")
    ap.add_argument("--out", default="badcase_out", help="输出目录")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    label_map = DEFAULT_LABEL_MAP

    gt_by_file, _, gt_drop = load_coco_by_filename(args.gt, label_map)
    vlm_by_file, _, vlm_drop = load_coco_by_filename(args.vlm, label_map)
    print(f"[load] GT  files={len(gt_by_file)}  (dropped {gt_drop} 个非核心类标注)")
    print(f"[load] VLM files={len(vlm_by_file)} (dropped {vlm_drop} 个映射不到核心类的标注)")

    # 文件名对齐检查 —— 三套必须落在同一批图上才有意义
    common = set(gt_by_file) & set(vlm_by_file)
    only_gt = set(gt_by_file) - set(vlm_by_file)
    if not common:
        print("[WARN] GT 与 VLM 没有任何同名图片!检查 file_name 是否一致(路径前缀/扩展名)。")
    print(f"[align] GT∩VLM 同名图 = {len(common)};仅 GT 有 = {len(only_gt)}")

    vlm_ev = evaluate(gt_by_file, vlm_by_file, args.iou)

    report = [f"# Badcase 报告 (IoU={args.iou})", ""]
    report.append(fmt_block("VLM 伪标注", vlm_ev))
    report.append("")

    result_json = {"iou": args.iou,
                   "vlm_vs_gt": {"global": vlm_ev["global"],
                                 "by_class": vlm_ev["by_class"],
                                 "bin_tp": vlm_ev["bin_tp"],
                                 "bin_total": vlm_ev["bin_total"]}}

    if args.pred:
        pred_by_file, _, pred_drop = load_coco_by_filename(args.pred, label_map)
        print(f"[load] PRED files={len(pred_by_file)} (dropped {pred_drop})")
        pred_ev = evaluate(gt_by_file, pred_by_file, args.iou)
        report.append(fmt_block("小模型 DINO-DETR", pred_ev))
        report.append("")
        resc = rescue_metric(gt_by_file, pred_ev, vlm_by_file, args.iou)
        report.append("### 杀手指标 — VLM 对小模型漏检的补救率")
        report.append(f"- 小模型漏检真值数 (pred FN) = {resc['pred_fn_total']}")
        report.append(f"- 其中被 VLM 补回 = {resc['rescued']}")
        report.append(f"- **rescue_rate = {resc['rescue_rate']:.3f}**  "
                      f"(VLM 辅助数据分析的核心价值)")
        report.append("")
        result_json["pred_vs_gt"] = {"global": pred_ev["global"],
                                     "by_class": pred_ev["by_class"],
                                     "bin_tp": pred_ev["bin_tp"],
                                     "bin_total": pred_ev["bin_total"]}
        result_json["rescue"] = {k: resc[k] for k in
                                 ("rescued", "pred_fn_total", "rescue_rate")}

    # badcase 排行(基于 VLM 漏检,小目标加权)
    ranking = badcase_ranking(gt_by_file, vlm_ev, args.top_n)
    result_json["badcase_top"] = ranking

    # 落盘
    report_md = os.path.join(args.out, "report.md")
    report_js = os.path.join(args.out, "result.json")
    badcase_txt = os.path.join(args.out, "badcase_images.txt")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    with open(report_js, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    with open(badcase_txt, "w", encoding="utf-8") as f:
        for r in ranking:
            f.write(f"{r['file_name']}\tscore={r['badcase_score']}\t"
                    f"fn={r['fn']}\tsmall_miss={r['small_miss']}\n")

    print("\n" + "\n".join(report))
    print(f"\n[OK] 报告 -> {report_md}")
    print(f"[OK] 指标 -> {report_js}")
    print(f"[OK] badcase 图清单 -> {badcase_txt} (top {len(ranking)})")


if __name__ == "__main__":
    main()
