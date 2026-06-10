#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_tile.py  ——  一次性诊断:为什么切图对某张图返回空。

对同一张图,用切图实际使用的 VISDRONE_PROMPT,分别在:
  (A) 整图
  (B) 每个 640px 切块
上跑推理,并打印**模型原始文本**(截断前 600 字)+ 解析出的框数。

目的:隔离"prompt 问题"还是"切块问题"。
  - 若整图(VISDRONE_PROMPT)也空、但 structured_vlm(JSON_PROMPT)非空 → prompt 的锅
  - 若整图非空、切块空 → 切块的锅

用法:
  python diag_tile.py --model /path/qwen2.5-vl-7b --image /tmp/one/xxx.jpg
"""

import argparse
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

from structured_vlm import load_model, resized_hw, extract_json, rescale_and_validate
from tiled_vlm import make_tiles, VISDRONE_PROMPT


def run(model, processor, pil_img, prompt, max_new_tokens=1024):
    """返回 (raw_text, dets)。"""
    orig_w, orig_h = pil_img.size
    res_h, res_w = resized_hw(processor, orig_h, orig_w)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_img},
        {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    raw_text = processor.batch_decode(trimmed, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]
    try:
        raw = extract_json(raw_text)
        dets = rescale_and_validate(raw, orig_h, orig_w, res_h, res_w)
    except Exception as e:
        dets = f"<parse error: {type(e).__name__}: {e}>"
    return raw_text, dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    model, processor = load_model(args.model)
    img = Image.open(args.image).convert("RGB")
    W, H = img.size
    print(f"\n===== IMAGE {args.image}  size={W}x{H} (WxH) =====")

    print("\n##### (A) WHOLE IMAGE with VISDRONE_PROMPT #####")
    raw, dets = run(model, processor, img, VISDRONE_PROMPT, args.max_new_tokens)
    n = len(dets) if isinstance(dets, list) else dets
    print(f"--- raw text (first 600 chars) ---\n{raw[:600]}")
    print(f"--- parsed: {n} detection(s) ---")

    tiles = make_tiles(W, H, 640, 0.2)
    print(f"\n##### (B) {len(tiles)} TILES @640px with VISDRONE_PROMPT #####")
    for ti, (x0, y0, x1, y1) in enumerate(tiles):
        crop = img.crop((x0, y0, x1, y1))
        raw, dets = run(model, processor, crop, VISDRONE_PROMPT, args.max_new_tokens)
        n = len(dets) if isinstance(dets, list) else dets
        print(f"\n--- tile {ti} ({x0},{y0},{x1},{y1}) size={x1-x0}x{y1-y0} ---")
        print(f"raw (first 400): {raw[:400]!r}")
        print(f"parsed: {n} detection(s)")


if __name__ == "__main__":
    main()
