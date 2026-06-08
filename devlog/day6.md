# Day 6 — End-to-end pipeline closure (VLM → COCO)

**Date:** ~2026-06-10
**Track:** 方向二 (VLM data analysis)
**Goal:** Close the full pipeline so the batch VLM output is directly consumable downstream — image folder → VLM → structured JSON → COCO format loadable by `pycocotools`.

## What I did

Connected the existing batch output (`annotations.json` from `structured_vlm.py`) to a
downstream-ready format by writing a conversion layer `to_coco.py`. The full chain is now:

```
images/ → structured_vlm.py (batch) → annotations.json → to_coco.py → coco.json → pycocotools.COCO() ✓
```

The conversion does three things: builds a category map, converts bbox format, and
writes a standard COCO detection JSON. I then verified the output loads cleanly with
`pycocotools` (instantiate `COCO()`, call `loadCats` / `loadAnns`), which is a harder
correctness bar than eyeballing the JSON.

## Pitfalls hit & fixed

1. **category_id reproducibility.** Dynamically scanning labels to build categories is
   convenient, but assigning ids in label-encounter order means the same class (e.g.
   `car`) can receive a different id on a different batch — silently breaking downstream
   class mapping. Fix: sort labels alphabetically before assigning ids, so the mapping is
   reproducible as long as the label set is the same.

2. **Empty-detection images must still be recorded.** An image with zero detections has
   to appear in the `images` field anyway, or the downstream image count won't match the
   annotation set. Verified this with a real empty case (one image returned `[]`) — it is
   correctly included in `images` with no corresponding annotations.

3. **bbox format conversion.** VLM outputs `[x1, y1, x2, y2]` in original-image pixels;
   COCO expects `[x, y, w, h]`. Added defensive clamping (min/max) in case the model ever
   emits flipped corners, so width/height can't go negative.

## Design decision

Kept a `score` field on each annotation. Strictly, COCO ground-truth annotations carry no
score (score belongs to prediction files). But this output is **VLM pseudo-labels, not
true GT** — for the Day 9 work (comparing VLM output against the small model to surface
disagreement/badcases), confidence is needed. So `score` stays, and the whole chain keeps
a consistent pseudo-label semantics rather than pretending to be GT.

## Data observation

Real run: 3 images, 8 detections. The bbox areas are small (hundreds to ~3600 px on
2000×1500 images), confirming the high small-object density typical of aerial scenes.
This flags the Day 9 badcase analysis as high-value — small objects are exactly where a
detector tends to miss.

## Output

- `to_coco.py` — conversion + verification (`--verify` runs the pycocotools load check)
- Verified end-to-end: `images=3, annotations=8, categories=3 (car/motorcycle/truck)`,
  `pycocotools` load passed.

## Next (Day 7)

Buffer + week-1 review: organize 方向二 code (`structured_vlm.py` + `to_coco.py`), first
push to GitHub, write the week-1 English summary.
