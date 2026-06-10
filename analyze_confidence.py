#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_confidence.py  ——  置信度阈值 → PR 曲线 / 最佳工作点

VLM 每个框带自报 confidence。整图/切图评估默认用所有框(等于阈值=0)。
本脚本扫一遍 confidence 阈值:在每个阈值上只保留 score>=t 的 VLM 框,
复用 badcase.py 的同口径匹配(file_name 对齐 + 核心类归一 + 贪心 IoU>=0.5),
算出 P/R/F1,据此:
  - 画 PR 曲线(recall-precision),保存 PNG;
  - 找 **最佳 F1 工作点**(以及"保 precision>=0.7 时的最大 recall"等实用点)。

纯 CPU、只读 COCO json。matplotlib 仅用于出图(无显示后端)。

用法:
  python analyze_confidence.py --gt gt.json --vlm vlm_coco.json --out conf_analysis/
"""

import argparse
import json
import os

from badcase import load_coco_by_filename, evaluate, prf, DEFAULT_LABEL_MAP


def filter_by_score(dets_by_file, thr):
    """只保留 score>=thr 的检测;GT 不动。"""
    out = {}
    for f, dets in dets_by_file.items():
        out[f] = [d for d in dets if d["score"] >= thr]
    return out


def sweep(gt_by_file, vlm_by_file, thresholds, iou_thr):
    rows = []
    for t in thresholds:
        filt = filter_by_score(vlm_by_file, t)
        ev = evaluate(gt_by_file, filt, iou_thr)
        g = ev["global"]
        p, r, f = prf(g["tp"], g["fp"], g["fn"])
        rows.append({"thr": round(t, 3), "P": round(p, 4), "R": round(r, 4),
                     "F1": round(f, 4), "tp": g["tp"], "fp": g["fp"], "fn": g["fn"]})
    return rows


def plot_pr(rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [x["R"] for x in rows]
    pres = [x["P"] for x in rows]
    thrs = [x["thr"] for x in rows]

    plt.figure(figsize=(7, 6))
    plt.plot(recs, pres, "-o", color="#2b6cb0", linewidth=2, markersize=5)
    for x in rows:
        plt.annotate(f"{x['thr']:.2f}", (x["R"], x["P"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=8)
    # 标出最佳 F1 点
    best = max(rows, key=lambda x: x["F1"])
    plt.scatter([best["R"]], [best["P"]], s=160, facecolors="none",
                edgecolors="#e53e3e", linewidths=2,
                label=f"best F1={best['F1']:.3f} @thr={best['thr']}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("VLM pseudo-label P-R vs confidence threshold (IoU=0.5)")
    plt.xlim(0, max(0.05, max(recs) * 1.15))
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Confidence-threshold PR analysis for VLM pseudo-labels")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--vlm", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--thresholds", default="0,0.5,0.6,0.7,0.8,0.85,0.9,0.95",
                    help="逗号分隔的阈值列表")
    ap.add_argument("--out", default="conf_analysis")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    lm = DEFAULT_LABEL_MAP
    gt_by_file, _, _ = load_coco_by_filename(args.gt, lm)
    vlm_by_file, _, _ = load_coco_by_filename(args.vlm, lm)

    thrs = [float(x) for x in args.thresholds.split(",") if x.strip() != ""]
    rows = sweep(gt_by_file, vlm_by_file, thrs, args.iou)

    # 实用工作点
    best_f1 = max(rows, key=lambda x: x["F1"])
    prec70 = [x for x in rows if x["P"] >= 0.70]
    best_r_at_p70 = max(prec70, key=lambda x: x["R"]) if prec70 else None

    # 落盘
    out_png = os.path.join(args.out, "pr_curve.png")
    plot_pr(rows, out_png)
    with open(os.path.join(args.out, "pr_table.json"), "w", encoding="utf-8") as f:
        json.dump({"iou": args.iou, "rows": rows,
                   "best_f1": best_f1,
                   "best_recall_at_precision>=0.70": best_r_at_p70}, f,
                  ensure_ascii=False, indent=2)

    # 打印
    print(f"{'thr':>5} {'P':>7} {'R':>7} {'F1':>7} {'TP':>6} {'FP':>6} {'FN':>6}")
    for x in rows:
        print(f"{x['thr']:>5} {x['P']:>7.3f} {x['R']:>7.3f} {x['F1']:>7.3f} "
              f"{x['tp']:>6} {x['fp']:>6} {x['fn']:>6}")
    print(f"\n[best F1]  thr={best_f1['thr']}  P={best_f1['P']:.3f} "
          f"R={best_f1['R']:.3f} F1={best_f1['F1']:.3f}")
    if best_r_at_p70:
        print(f"[P>=0.70]  thr={best_r_at_p70['thr']}  P={best_r_at_p70['P']:.3f} "
              f"R={best_r_at_p70['R']:.3f}")
    print(f"\n[OK] PR 曲线 -> {out_png}")
    print(f"[OK] 数据表 -> {os.path.join(args.out, 'pr_table.json')}")


if __name__ == "__main__":
    main()
