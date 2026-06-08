# VLM Auto-Annotator

A small, progressive pipeline that turns images into structured JSON
annotations using Qwen2.5-VL. For each image it produces `label` + `bbox` +
`confidence` detections, ready for downstream training, evaluation, or bad-case
triage.

The goal is a **reliable end-to-end link**, not maximum accuracy: load the
model once, batch over a folder, tolerate per-image failures, and emit one
aggregated JSON with run-level statistics.

## Project layout

```
vlm_auto_annotator/
├── minimal_vlm.py      # Day 1 — single image, free-form text output
├── structured_vlm.py   # Day 2+4 — main pipeline: structured JSON, bbox rescale, batch export
├── to_coco.py          # Day 6 — convert batch JSON to COCO (pycocotools-verified)
├── requirements.txt
├── devlog/             # daily notes (environment, bugs, design decisions)
│   ├── day1_2026-06-05.md
│   ├── day2.md
│   ├── day3.md         # DINO-DETR / ONNX track (separate small-model path)
│   ├── day4.md
│   └── day6.md         # end-to-end VLM -> COCO closure
└── README.md
```

Scripts build on each other:

| Script | What it adds |
|--------|--------------|
| `minimal_vlm.py` | Load Qwen2.5-VL, run one image, print description + latency/VRAM |
| `structured_vlm.py` | Force JSON detections (multi-layer `extract_json` + validation); rescale bboxes to original pixels; batch a folder with per-image fault tolerance |
| `to_coco.py` | Convert the batch JSON to a standard COCO detection file; verify it loads with `pycocotools` |

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

## Dev log

See `devlog/` for day-by-day notes: environment setup, parsing pitfalls, bbox
rescaling verification, batch hardening, the end-to-end COCO closure (Day 6),
and (Day 3) the parallel DINO-DETR / ONNX export track on VisDrone.

## Notes & limitations

- A general VLM recalls far fewer objects than a dedicated dense detector on
  cluttered scenes. This pipeline is meant for **auto-labeling and bad-case
  triage**, not as a replacement for a trained detector.
- `confidence` is the model's self-reported certainty, not a calibrated score.
- On dense images, raise `--max-new-tokens` (e.g. 4096) to reduce truncated JSON.

## License

Apache-2.0.
