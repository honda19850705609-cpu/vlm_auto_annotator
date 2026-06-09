#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tiled_vlm.py  ——  D11 改进:切图(tiling)推理,绕开 token 预算硬顶

动机
----
badcase.py 量出 VLM 在 VisDrone 上召回仅 ~4.5%,且随目标变小急剧崩塌。
原因有二:
  (a) 显著性偏好:VLM 只标显眼的大目标;
  (b) token 预算硬顶:整图一次推理,1024~4096 token 根本吐不下密集小目标
      (一张图 300+ 目标,JSON 物理上写不完)。

切图正面解决 (b),并部分缓解 (a):把整图切成若干带重叠的小块,
每块单独喂 VLM。小块里目标少、且每个目标在块内相对更"大",
VLM 既不会撞 token 顶,也更容易看见小目标。框最后拼回原图坐标、
跨块去重(NMS),输出与 structured_vlm.py 完全一致的 annotations.json,
因此 to_coco.py / badcase.py 无需任何改动即可复用。

链路
----
  图像文件夹 -> [本脚本: 切图+逐块VLM+拼回+NMS] -> annotations_tiled.json
            -> to_coco.py -> vlm_tiled_coco.json -> badcase.py 对照 GT

用法
----
  python tiled_vlm.py \
      --model ./qwen2.5-vl-7b \
      --image-dir ./visdrone_val_gt109 \
      --out ./Day9/annotations_tiled.json \
      --tile-size 640 --overlap 0.2 --nms-iou 0.55 --max-new-tokens 1024

成本提示:每图被切成 (行x列) 块,VLM 调用数 ≈ 图数 × 块数。
640px 切图在 1920x1080 上约 3x3=9 块,109 图 ≈ ~1000 次调用。
比整图慢一个量级,属正常;显存紧可换 ./qwen2.5-vl-3b。
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

# 复用 structured_vlm.py 里已验证过的部件,避免重造轮子。
from structured_vlm import (
    load_model,
    resized_hw,
    extract_json,
    rescale_and_validate,
    IMG_EXTS,
)


# ---------------------------------------------------------------------------
# v2 改进:类别约束 + 反幻觉 prompt(替代 structured_vlm 的通用 JSON_PROMPT)。
# smoke 暴露的问题:VLM 标 building/road/trees 等场景元素、把整块当成一个
# 巨大的 "pedestrian/car"、还编出等距的竖列假框。这里从源头掐住。
# ---------------------------------------------------------------------------
VISDRONE_PROMPT = (
    "This is an aerial / drone-view image. Detect EVERY individual small object. "
    "Use ONLY these exact labels: "
    "pedestrian, person, car, van, truck, bus, bicycle, tricycle, motor. "
    "Do NOT label scenery such as building, road, street, tree, sky, parking lot, "
    "billboard — ignore them completely. "
    "Each bounding box must tightly enclose exactly ONE object. "
    "Do NOT merge several objects into one big box. "
    "Do NOT invent evenly-spaced or repeated boxes. "
    "Respond with ONLY a JSON array, no prose, no markdown fences. "
    "Each element: "
    '{"label": one of the allowed labels, '
    '"bbox": [x1, y1, x2, y2] in absolute pixel integers, '
    '"confidence": 0.0-1.0}. '
    "If you see no such object, respond with []."
)

# 即使模型不听话,也用白名单兜底过滤掉非目标类(含简单的复数归一)。
ALLOWED_LABELS = {
    "pedestrian", "person", "people", "car", "van", "truck", "bus",
    "bicycle", "bike", "tricycle", "motor", "motorcycle", "motorbike",
}


def normalize_label(label):
    """小写、去复数;返回白名单内的标签,否则 None(=丢弃)。"""
    s = str(label).strip().lower()
    if s in ALLOWED_LABELS:
        return s
    if s.endswith("s") and s[:-1] in ALLOWED_LABELS:  # cars -> car
        return s[:-1]
    return None


# ---------------------------------------------------------------------------
# 切图:沿每个轴生成起点,保证全覆盖 + 末尾贴边 + 带重叠。
# ---------------------------------------------------------------------------
def axis_starts(length, tile, overlap):
    """返回某一轴上所有 tile 的起点坐标列表(含末尾贴边块)。"""
    tile = min(tile, length)
    stride = max(1, int(round(tile * (1.0 - overlap))))
    if tile >= length:
        return [0], tile
    starts = list(range(0, length - tile + 1, stride))
    if not starts:
        starts = [0]
    if starts[-1] != length - tile:
        starts.append(length - tile)  # 贴右/下边,杜绝边缘漏覆盖
    return starts, tile


def make_tiles(W, H, tile_size, overlap):
    """生成 (x0, y0, x1, y1) 块列表。"""
    xs, tw = axis_starts(W, tile_size, overlap)
    ys, th = axis_starts(H, tile_size, overlap)
    tiles = []
    for y0 in ys:
        for x0 in xs:
            tiles.append((x0, y0, x0 + tw, y0 + th))
    return tiles


# ---------------------------------------------------------------------------
# 跨块去重:同一目标会在重叠区被多块各标一次,用按类别的贪心 NMS 合并。
# ---------------------------------------------------------------------------
def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def nms_per_label(dets, iou_thr):
    """按 label 分组,组内按 confidence 降序贪心抑制。返回去重后的列表。"""
    by_label = {}
    for d in dets:
        by_label.setdefault(d["label"], []).append(d)

    kept_all = []
    for label, group in by_label.items():
        group.sort(key=lambda d: d.get("confidence", 0.0), reverse=True)
        kept = []
        for d in group:
            if all(_iou_xyxy(d["bbox"], k["bbox"]) < iou_thr for k in kept):
                kept.append(d)
        kept_all.extend(kept)
    return kept_all


# ---------------------------------------------------------------------------
# 单块推理:与 structured_vlm.infer_one 同逻辑,但吃 PIL 子图而非路径。
# ---------------------------------------------------------------------------
def infer_pil(model, processor, pil_img, prompt, max_new_tokens=1024):
    orig_w, orig_h = pil_img.size
    res_h, res_w = resized_hw(processor, orig_h, orig_w)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    raw_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    try:
        raw = extract_json(raw_text)
    except ValueError:
        return []
    # rescale_and_validate 把模型空间映射回"这块子图"的像素坐标
    return rescale_and_validate(raw, orig_h, orig_w, res_h, res_w)


def infer_one_tiled(model, processor, image_path, tile_size, overlap,
                    nms_iou, max_new_tokens, upscale=2.0, max_box_frac=0.5,
                    debug=False):
    """对一张整图做切图推理,返回拼回原图坐标 + 去重后的检测列表。

    v2 三项改进:
      - upscale:每个切块先放大 upscale 倍再喂,等效再放大微小目标;
                 模型空间坐标先除以 upscale 还原到块坐标,再加块偏移。
      - max_box_frac:丢弃面积 > 块面积 * max_box_frac 的框(干掉"整块当一物"的巨框)。
      - normalize_label:白名单过滤,非 VisDrone 目标类直接丢。
    """
    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    tiles = make_tiles(W, H, tile_size, overlap)

    n_label_drop = n_big_drop = 0
    all_dets = []
    for (x0, y0, x1, y1) in tiles:
        tw, th = x1 - x0, y1 - y0
        crop = image.crop((x0, y0, x1, y1))
        if upscale and upscale != 1.0:
            crop_fed = crop.resize((max(1, int(tw * upscale)),
                                    max(1, int(th * upscale))))
        else:
            crop_fed = crop
        dets = infer_pil(model, processor, crop_fed, VISDRONE_PROMPT, max_new_tokens)

        tile_area = float(tw * th)
        for d in dets:
            label = normalize_label(d["label"])
            if label is None:                      # 非目标类:丢
                n_label_drop += 1
                continue
            bx1, by1, bx2, by2 = d["bbox"]
            # 放大空间 -> 块空间
            if upscale and upscale != 1.0:
                bx1, by1, bx2, by2 = (bx1 / upscale, by1 / upscale,
                                      bx2 / upscale, by2 / upscale)
            if (bx2 - bx1) * (by2 - by1) > max_box_frac * tile_area:  # 巨框:丢
                n_big_drop += 1
                continue
            # 块空间 -> 全图空间
            all_dets.append({
                "label": label,
                "bbox": [round(bx1 + x0, 1), round(by1 + y0, 1),
                         round(bx2 + x0, 1), round(by2 + y0, 1)],
                "confidence": d.get("confidence", 0.0),
            })

    merged = nms_per_label(all_dets, nms_iou)
    if debug:
        print(f"    tiles={len(tiles)}  kept={len(all_dets)}  "
              f"(label_drop={n_label_drop} big_drop={n_big_drop})  after_nms={len(merged)}")
    return merged


def run(args, model, processor):
    img_dir = Path(args.image_dir)
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f">>> no images found in {img_dir}")
        return
    print(f">>> found {len(images)} images; tile={args.tile_size} overlap={args.overlap} "
          f"nms_iou={args.nms_iou} upscale={args.upscale} max_box_frac={args.max_box_frac}")

    results = {}
    n_ok = n_fail = n_dets = 0
    t0 = time.time()
    for i, path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {path.name}")
        try:
            dets = infer_one_tiled(
                model, processor, str(path),
                tile_size=args.tile_size, overlap=args.overlap,
                nms_iou=args.nms_iou, max_new_tokens=args.max_new_tokens,
                upscale=args.upscale, max_box_frac=args.max_box_frac,
                debug=args.debug,
            )
            results[path.name] = dets
            n_ok += 1
            n_dets += len(dets)
            print(f"    -> {len(dets)} detection(s)")
        except Exception as e:  # 一张坏图不能拖垮整批
            results[path.name] = {"error": f"{type(e).__name__}: {e}"}
            n_fail += 1
            print(f"    [FAIL] {type(e).__name__}: {e}")

    payload = {
        "model": args.model,
        "mode": "tiled",
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "nms_iou": args.nms_iou,
        "upscale": args.upscale,
        "max_box_frac": args.max_box_frac,
        "num_images": len(images),
        "num_ok": n_ok,
        "num_failed": n_fail,
        "total_detections": n_dets,
        "elapsed_sec": round(time.time() - t0, 1),
        "annotations": results,  # 与 structured_vlm 同格式,to_coco.py 可直接吃
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\n===== TILED BATCH DONE =====")
    print(f"ok={n_ok}  failed={n_fail}  total_detections={n_dets}  "
          f"elapsed={payload['elapsed_sec']}s")
    print(f">>> written: {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Tiled VLM inference to beat the token ceiling.")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="local model dir or HuggingFace repo id")
    p.add_argument("--image-dir", required=True, help="folder of images")
    p.add_argument("--out", default="annotations_tiled.json", help="output JSON path")
    p.add_argument("--tile-size", type=int, default=512,
                   help="tile edge in px (default 512; smaller = more magnification)")
    p.add_argument("--overlap", type=float, default=0.2,
                   help="fractional tile overlap 0~0.5 (default 0.2)")
    p.add_argument("--nms-iou", type=float, default=0.55,
                   help="cross-tile dedupe IoU threshold (default 0.55)")
    p.add_argument("--upscale", type=float, default=2.0,
                   help="enlarge each tile by this factor before inference (default 2.0; "
                        "1.0 disables). Magnifies tiny objects for the model.")
    p.add_argument("--max-box-frac", type=float, default=0.5,
                   help="drop boxes larger than this fraction of the tile area "
                        "(default 0.5; kills whole-region mislabels)")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--debug", action="store_true",
                   help="print per-image tile / raw / nms counts")
    return p.parse_args()


def main():
    args = parse_args()
    model, processor = load_model(args.model)
    run(args, model, processor)


if __name__ == "__main__":
    main()
