# VLM Auto-Annotator

A small, progressive pipeline that turns images into structured JSON
annotations using Qwen2.5-VL. For each image it produces `label` + `bbox` +
`confidence` detections, ready for downstream training, evaluation, or bad-case
triage.

The goal is a **reliable end-to-end link**, not maximum accuracy: load the
model once, batch over a folder, tolerate per-image failures, and emit one
aggregated JSON with run-level statistics.

## Key result (VisDrone-DET val, 109 images vs human GT)

Quantifying VLM auto-annotation quality against ground truth, then fixing its
weakest regime (dense small objects) with multi-scale tiling:

| | integral VLM | **multi-scale tiling** |
|---|---|---|
| recall | 0.045 | **0.243** (5.4×) |
| F1 | 0.082 | **0.336** (4.1×) |
| precision | 0.538 | **0.544** (flat — no cost) |

The gain is **inversely proportional to object size** (≥96px: 1.5× → 8–16px:
**20×**) — tiling adds recall exactly where whole-image inference collapses, at
no precision cost. Full method, the v1→v2→v3 ablation, and honest limits in
[`devlog/day9.md`](devlog/day9.md); raw reports in [`results/`](results/).

## Project layout

```
vlm_auto_annotator/
├── minimal_vlm.py      # Day 1 — single image, free-form text output
├── structured_vlm.py   # Day 2+4 — main pipeline: structured JSON, bbox rescale, batch export
├── to_coco.py          # Day 6 — convert batch JSON to COCO (pycocotools-verified)
├── badcase.py          # Day 9 — GT-anchored eval: P/R/F1 + per-size recall + badcase ranking
├── tiled_vlm.py        # Day 11 — multi-scale tiling inference (beats the token ceiling)
├── requirements.txt
├── results/            # full-val eval reports (integral baseline vs tiled v3)
├── devlog/             # daily notes (environment, bugs, design decisions)
│   ├── day1_2026-06-05.md
│   ├── day2.md
│   ├── day3.md         # DINO-DETR / ONNX track (separate small-model path)
│   ├── day4.md
│   ├── day6.md         # end-to-end VLM -> COCO closure
│   └── day9.md         # GT-anchored badcase + tiling improvement (v1/v2/v3 ablation)
└── README.md
```

Scripts build on each other:

| Script | What it adds |
|--------|--------------|
| `minimal_vlm.py` | Load Qwen2.5-VL, run one image, print description + latency/VRAM |
| `structured_vlm.py` | Force JSON detections (multi-layer `extract_json` + validation); rescale bboxes to original pixels; batch a folder with per-image fault tolerance |
| `to_coco.py` | Convert the batch JSON to a standard COCO detection file; verify it loads with `pycocotools` |
| `badcase.py` | Score VLM pseudo-labels against ground truth (file-name aligned, label-normalized, greedy IoU): overall/per-class P/R/F1, **recall split by object size**, ranked bad-case image list |
| `tiled_vlm.py` | Multi-scale tiling inference: split each image into overlapping tiles (optionally upscaled), detect per tile, map boxes back to global coords, cross-tile NMS — recovers small-object recall that whole-image inference loses |

Use **`structured_vlm.py`** for production-style auto-annotation, then
**`to_coco.py`** to hand the result to downstream COCO tooling. `minimal_vlm.py`
is kept as a stepping stone and for debugging.

## What it does

```
image folder  ->  Qwen2.5-VL  ->  structured JSON  ->  COCO  ->  downstream / triage
```

- **Structured output.** Prompts the model for a strict JSON array
  (`label`, `bbox`, `confidence`) and parses it, instead of free-form text.
- **Correct bounding boxes.** Qwen2.5-VL emits coordinates in its internal
  `smart_resize` pixel space, *not* the original image resolution. Boxes are
  rescaled back to original pixels, then ordered and clamped to image bounds.
- **Truncation-robust parsing.** On dense scenes the JSON output can be cut off
  by `max_new_tokens` mid-object. The parser salvages every complete `{...}`
  object and drops the trailing partial one instead of crashing.
- **Batch with fault tolerance.** The model loads once and is reused; each image
  runs in its own `try/except`, so a single unreadable/corrupt file is recorded
  as an error and skipped rather than killing the whole run.

## Install

```bash
pip install -r requirements.txt
```

Pinned versions: `transformers==5.9.0`, `qwen-vl-utils==0.0.14`.

Tested on RTX 4080 SUPER / RTX 5090 with the **7B** model in bf16. The 7B model
is the practical minimum for usable grounding; the 3B variant tends to return
empty results on dense scenes.

Download weights locally (ModelScope or HuggingFace) or pass a HuggingFace repo
id to `--model`. Model weight directories are gitignored.

## Usage

### Day 1 — sanity check (free-form text)

```bash
python minimal_vlm.py \
    --model /path/to/qwen2.5-vl-7b \
    --image /path/to/test.jpg
```

### Main pipeline — structured JSON (recommended)

Single image first, to visually verify boxes are placed correctly:

```bash
python structured_vlm.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --image /path/to/test.jpg \
    --visualize \
    --max-new-tokens 4096
```

This prints the parsed detections and writes `test.annotated.jpg` next to the
input with boxes drawn on it, so you can eyeball that the rescaling is correct.

Then a whole folder:

```bash
python structured_vlm.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --image-dir /path/to/images/ \
    --out annotations.json \
    --max-new-tokens 4096
```

`--model` accepts either a local model directory or a HuggingFace repo id.
Use `--debug` in batch mode to print per-image resize info and raw model text.

### Convert to COCO

Turn the batch output into a standard COCO detection file and verify it loads:

```bash
python to_coco.py \
    --in annotations.json \
    --images /path/to/images/ \
    --out coco.json \
    --verify
```

`--verify` runs a `pycocotools` load check (instantiates `COCO()`, reads back
categories / images / annotations) as an end-to-end correctness bar.

## Output format

Batch mode (`structured_vlm.py --image-dir`) writes:

```json
{
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "num_images": 3,
  "num_ok": 3,
  "num_failed": 0,
  "total_detections": 8,
  "elapsed_sec": 8.5,
  "annotations": {
    "img1.jpg": [
      { "label": "car", "bbox": [463.9, 256.8, 497.1, 291.3], "confidence": 0.9 }
    ],
    "img2.jpg": [ ... ],
    "img3.jpg": []
  }
}
```

`bbox` is `[x1, y1, x2, y2]` in original-image pixels. An empty list means the
model returned no detections for that image — a natural bad-case candidate.
A failed image appears as `{"error": "..."}` under its filename.

## Results — how good are the pseudo-labels, and can we improve them?

Evaluated on VisDrone DET-val (109 images, 7971 core-class GT boxes) with `badcase.py`
(IoU≥0.5, classes merged to pedestrian / vehicle / bicycle).

**Whole-image VLM auto-annotation is unreliable on dense aerial data** — and the
failure is concentrated in small objects:

| recall by object size | <8px | 8–16px | 16–32px | 32–96px | ≥96px |
|---|---|---|---|---|---|
| whole-image VLM | 0.004 | 0.005 | 0.024 | 0.092 | 0.324 |

Overall: **P=0.538, R=0.045, F1=0.082** — it misses ~95% of objects.

**Tiling (`tiled_vlm.py`, multi-scale 640px + 512px@2×) recovers most of the loss
without costing precision:**

| metric | whole-image | tiled (multi-scale) | gain |
|---|---|---|---|
| precision | 0.538 | 0.544 | flat |
| recall | 0.045 | **0.243** | **5.4×** |
| F1 | 0.082 | **0.336** | 4.1× |
| recall 8–16px | 0.005 | 0.100 | **20×** |
| recall 16–32px | 0.024 | 0.255 | **10.6×** |

The gain is largest for the smallest objects (20× at 8–16px, 1.5× at ≥96px): tiling
raises effective resolution per object and sidesteps the output token ceiling, adding
recall exactly where whole-image inference collapses — while a class whitelist keeps
precision flat. **<8px objects remain a hard floor** (0.004 → 0.015): below the VLM's
post-`smart_resize` resolution, they are largely beyond reach regardless of tiling.

Takeaway: use VLM pseudo-labels for **auto-labeling + bad-case triage with tiling**,
and treat sub-8px objects as a known blind spot. See `devlog/day9.md` for the full
v1/v2/v3 ablation.

### Usage — eval & tiling

```bash
# Score VLM output against ground truth
python badcase.py --gt gt_coco.json --vlm vlm_coco.json --out badcase_out/

# Multi-scale tiled inference (default scales: 640:1.0,512:2.0)
python tiled_vlm.py --model Qwen/Qwen2.5-VL-7B-Instruct \
    --image-dir images/ --out annotations_tiled.json \
    --scales 640:1.0,512:2.0 --overlap 0.2 --nms-iou 0.55
```

## Dev log

See `devlog/` for day-by-day notes: environment setup, parsing pitfalls, bbox
rescaling verification, batch hardening, the end-to-end COCO closure (Day 6),
and (Day 3) the parallel DINO-DETR / ONNX export track on VisDrone.

## Notes & limitations

- A general VLM recalls far fewer objects than a dedicated dense detector on
  cluttered scenes. This pipeline is meant for **auto-labeling and bad-case
  triage**, not as a replacement for a trained detector. Tiling narrows the gap
  (recall 0.045 → 0.243) but does not close it.
- **Sub-8px objects are a hard floor** (recall ~0.015 even with tiling): they fall
  below the model's `smart_resize` resolution and are largely unrecoverable.
- Even with tiling, the densest tiles can still overflow `max_new_tokens` (the parser
  salvages complete objects); the token ceiling is fundamental, not fully solved.
- `confidence` is the model's self-reported certainty, not a calibrated score.
- On dense images, raise `--max-new-tokens` (e.g. 4096) to reduce truncated JSON.

## License

Apache-2.0.
