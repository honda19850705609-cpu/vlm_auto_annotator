"""Structured VLM inference: single-image JSON (D2) + folder batch (D4).

Builds on minimal_vlm.py (Day 1). Two things are added here:

  D2 -- prompt the model to emit structured JSON (label + bbox + confidence),
        robustly parse it, and rescale boxes back to the ORIGINAL image size.
  D4 -- loop over a folder of images, reuse one loaded model, tolerate
        per-image failures, and write one aggregated annotation file.

Key correctness point (the bbox trap):
    Qwen2.5-VL does NOT see your original image. The processor resizes it to
    a (smart_resize) grid before the model runs, and the coordinates the model
    emits are in THAT resized pixel space -- not your original resolution.
    So we compute the resized (h, w) the same way the processor does, then
    rescale every box by (orig / resized). Skipping this gives boxes that look
    plausible but are systematically wrong.

Usage -- single image first, to eyeball the coordinates:
    python structured_vlm.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --image samples/test.jpg \
        --visualize

Then a whole folder:
    python structured_vlm.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --image-dir samples/ \
        --out annotations.json \
        --max-new-tokens 4096
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# smart_resize moved around across qwen-vl-utils versions; try both, then fall
# back to a local copy of the official implementation so the script never dies
# on an import. The math is identical to the published version.
try:
    from qwen_vl_utils import smart_resize  # newer
except ImportError:
    try:
        from qwen_vl_utils.vision_process import smart_resize  # older
    except ImportError:
        smart_resize = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Qwen2.5-VL default vision config. If your processor reports different values
# we read them off the processor at runtime (see resized_hw); these are only a
# fallback.
_DEFAULT_MIN_PIXELS = 4 * 28 * 28
_DEFAULT_MAX_PIXELS = 16384 * 28 * 28
_PATCH = 28


def _smart_resize_fallback(h, w, factor=_PATCH,
                           min_pixels=_DEFAULT_MIN_PIXELS,
                           max_pixels=_DEFAULT_MAX_PIXELS):
    """Local copy of Qwen's smart_resize, used only if the import failed."""
    import math
    if max(h, w) / min(h, w) > 200:
        raise ValueError("aspect ratio too extreme for smart_resize")
    hb = max(factor, round(h / factor) * factor)
    wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


# The instruction that makes D2 work. Explicit schema, explicit coordinate
# space, "JSON only" to suppress the chatty preamble that breaks parsing.
JSON_PROMPT = (
    "Detect every distinct object in this image. "
    "Respond with ONLY a JSON array, no prose, no markdown fences. "
    "Each element must be an object with exactly these keys:\n"
    '  "label": a short class name (string)\n'
    '  "bbox": [x1, y1, x2, y2] in absolute pixel coordinates (integers)\n'
    '  "confidence": your certainty from 0.0 to 1.0 (float)\n'
    'Example: [{"label": "car", "bbox": [10, 20, 110, 90], "confidence": 0.95}]\n'
    "If you see no objects, respond with []."
)


def resized_hw(processor, orig_h, orig_w):
    """Return the (h, w) the model actually sees, matching the processor."""
    ip = getattr(processor, "image_processor", None)
    min_px = getattr(ip, "min_pixels", _DEFAULT_MIN_PIXELS) if ip else _DEFAULT_MIN_PIXELS
    max_px = getattr(ip, "max_pixels", _DEFAULT_MAX_PIXELS) if ip else _DEFAULT_MAX_PIXELS
    fn = smart_resize or _smart_resize_fallback
    try:
        return fn(orig_h, orig_w, factor=_PATCH, min_pixels=min_px, max_pixels=max_px)
    except TypeError:
        # some versions don't accept the kwargs
        return fn(orig_h, orig_w)


def extract_json(text):
    """Pull a JSON array out of the model's raw text. Returns a list.

    Handles the common failure where the model's output is truncated by
    max_new_tokens mid-object: we fall back to extracting every COMPLETE
    {...} object and drop the trailing partial one.
    """
    cleaned = text.strip()
    # strip ```json ... ``` fences if the model added them
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # 1) happy path: the whole thing is valid JSON
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # 2) grab the outermost [...] and try again
    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    # 3) truncated output: salvage every complete {...} object
    objs = []
    for obj_match in re.finditer(r"\{[^{}]*\}", cleaned):
        try:
            obj = json.loads(obj_match.group(0))
            if isinstance(obj, dict):
                objs.append(obj)
        except json.JSONDecodeError:
            continue
    if objs:
        print(f"    [warn] output looked truncated; salvaged {len(objs)} complete object(s)")
        return objs

    raise ValueError("no JSON array or objects found in model output")


def rescale_and_validate(raw, orig_h, orig_w, res_h, res_w):
    """Map model-space boxes back to original pixels; drop malformed entries."""
    sx = orig_w / res_w
    sy = orig_h / res_h
    out = []
    for i, det in enumerate(raw):
        try:
            label = str(det["label"])
            x1, y1, x2, y2 = det["bbox"]
            conf = float(det.get("confidence", 0.0))
            # rescale model-space -> original
            x1, x2 = x1 * sx, x2 * sx
            y1, y2 = y1 * sy, y2 * sy
            # order + clamp to image bounds
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            x1 = max(0, min(x1, orig_w))
            x2 = max(0, min(x2, orig_w))
            y1 = max(0, min(y1, orig_h))
            y2 = max(0, min(y2, orig_h))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue  # degenerate box
            out.append({
                "label": label,
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "confidence": round(conf, 3),
            })
        except (KeyError, ValueError, TypeError) as e:
            print(f"    [warn] skipped malformed detection #{i}: {e}")
            continue
    return out


def load_model(model_path):
    print(f">>> loading model from: {model_path}")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print(f">>> model loaded in {time.time() - t0:.1f}s")
    return model, processor


def infer_one(model, processor, image_path, max_new_tokens=1024, debug=False):
    """Run structured inference on one image. Returns a list of detections."""
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    res_h, res_w = resized_hw(processor, orig_h, orig_w)
    if debug:
        print(f"    orig (hxw): {orig_h}x{orig_w}  ->  model sees: {res_h}x{res_w}"
              f"  (scale x{orig_w/res_w:.3f}, y{orig_h/res_h:.3f})")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": JSON_PROMPT},
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
    if debug:
        print(f"    raw model text: {raw_text[:300]}")

    raw = extract_json(raw_text)
    return rescale_and_validate(raw, orig_h, orig_w, res_h, res_w)


def visualize(image_path, detections, out_path):
    """Draw boxes so you can eyeball whether the rescaling is correct."""
    from PIL import ImageDraw
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 12)), f"{d['label']} {d['confidence']}", fill="red")
    image.save(out_path)
    print(f">>> annotated image saved: {out_path}")


def run_single(args, model, processor):
    t0 = time.time()
    dets = infer_one(model, processor, args.image, args.max_new_tokens, debug=True)
    print(f"\n===== {len(dets)} detection(s) in {time.time()-t0:.1f}s =====")
    print(json.dumps(dets, indent=2, ensure_ascii=False))
    if args.visualize:
        out_img = Path(args.image).with_suffix(".annotated.jpg")
        visualize(args.image, dets, out_img)


def run_batch(args, model, processor):
    img_dir = Path(args.image_dir)
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f">>> no images found in {img_dir}")
        return
    print(f">>> found {len(images)} images")

    results = {}
    n_ok = n_fail = n_dets = 0
    t0 = time.time()
    for i, path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {path.name}")
        try:
            dets = infer_one(model, processor, str(path), args.max_new_tokens,
                             debug=args.debug)
            results[path.name] = dets
            n_ok += 1
            n_dets += len(dets)
            print(f"    -> {len(dets)} detection(s)")
        except Exception as e:  # one bad image must not kill the batch
            results[path.name] = {"error": f"{type(e).__name__}: {e}"}
            n_fail += 1
            print(f"    [FAIL] {type(e).__name__}: {e}")

    payload = {
        "model": args.model,
        "num_images": len(images),
        "num_ok": n_ok,
        "num_failed": n_fail,
        "total_detections": n_dets,
        "elapsed_sec": round(time.time() - t0, 1),
        "annotations": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\n===== BATCH DONE =====")
    print(f"ok={n_ok}  failed={n_fail}  total_detections={n_dets}  "
          f"elapsed={payload['elapsed_sec']}s")
    print(f">>> written: {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Structured VLM JSON + batch inference.")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="local model dir or HuggingFace repo id")
    p.add_argument("--image", help="single-image mode: path to one image")
    p.add_argument("--image-dir", help="batch mode: folder of images")
    p.add_argument("--out", default="annotations.json",
                   help="batch output JSON path")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--visualize", action="store_true",
                   help="single-image mode: draw boxes to verify rescaling")
    p.add_argument("--debug", action="store_true",
                   help="batch mode: print per-image resize / raw text")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.image and not args.image_dir:
        raise SystemExit("provide --image (single) or --image-dir (batch)")
    model, processor = load_model(args.model)
    if args.image_dir:
        run_batch(args, model, processor)
    else:
        run_single(args, model, processor)


if __name__ == "__main__":
    main()
