# Day 9 — GT-anchored badcase analysis + tiling improvement (VLM pseudo-labels)

**Date:** ~2026-06-09
**Track:** 方向二 (VLM data analysis)
**Goal (D9 + D11):** Quantify *how good/bad* the VLM's auto-annotations actually are by
anchoring them to VisDrone human ground truth, surface the worst cases, then try to
**improve** the weakest regime (dense small objects).

## Pipeline

```
VisDrone val images → VLM (integral OR tiled) → annotations.json
   → to_coco.py → coco.json → badcase.py  vs  GT (instances_val..._from_val.json)
```

`badcase.py` is GT-anchored and pure-CPU (no pycocotools): align by **file_name** (the
three JSONs use different image_id schemes), normalize free-text VLM labels into the
VisDrone core classes (pedestrian / vehicle / bicycle), then greedy per-image-per-class
IoU matching (IoU≥0.5 = hit). It reports overall P/R/F1, per-class, and **recall split by
object pixel size** — the size split is the whole point, since small objects are the命门.

## D9 — baseline finding (integral VLM, full 109-image val)

Integral (whole-image) VLM auto-annotation is **near-useless as labels on this data**:

- Overall recall ≈ **0.045** — it misses ~95% of true objects.
- Recall collapses monotonically with object size: large objects are found, sub-32px are
  almost entirely missed.

Two root causes, confirmed by inspecting the raw output:
1. **Token ceiling.** A dense aerial frame has 200–400 objects; the JSON for that many
   boxes physically overflows `max_new_tokens`, so output is truncated.
2. **Resolution bottleneck.** The processor's `smart_resize` downsamples the frame before
   the model sees it, so an 8px object becomes ~sub-pixel — physically invisible.

This is itself a clean, reportable result: **VLM pseudo-labels cannot be trusted for
dense small-object aerial data without intervention.**

## D11 — improvement: tiling, ablated over 3 versions

Tiling attacks both root causes: cut the frame into overlapping tiles, run the VLM per
tile (fewer objects → within token budget; higher effective resolution per object), map
boxes back to global coords, cross-tile NMS dedupe (`tiled_vlm.py`). Output format is
identical to `structured_vlm.py`, so `to_coco.py` / `badcase.py` are reused unchanged.

Evaluated fairly on the **5 densest images** (the integral-VLM badcase top-5, where it
scored ~0 TP; 1110 GT boxes after core-class filtering):

| metric | v1 (640px) | v2 (512px + 2× upscale, class-prompt, filters) | **v3 (multi-scale union)** |
|---|---|---|---|
| overall recall | 0.083 | 0.072 | **0.118** (+42% vs v1) |
| overall F1 | 0.141 | 0.122 | **0.183** |
| TP / FP | 92 / 114 | 79 / 110 | **130** / 186 |
| precision | 0.447 | 0.418 | 0.411 |
| recall <8px | 0.000 | 0.000 | 0.000 |
| recall 8–16px | 0.018 | 0.058 | 0.053 |
| recall 16–32px | 0.060 | 0.134 | **0.219** (3.6× v1) |
| recall 32–96px | **0.281** | 0.039 | 0.140 |
| pedestrian recall | 0.012 | 0.141 | **0.165** |
| vehicle recall | 0.156 | 0.034 | 0.101 |

**Reading the ablation:**
- **v1** (plain 640px tiling): rescues medium objects (32–96px: 0→0.281) but barely moves
  the small bins, and the generic prompt produced scenery labels (building/road/tree) and
  whole-region giant mislabels.
- **v2** (512px + 2× per-tile upscale + VisDrone-class prompt + label whitelist + giant-box
  filter): upscaling **tripled** the small bins (8–16: 0.018→0.058, 16–32: 0.060→0.134) and
  cleaned labels — but smaller tiles collapsed the medium bin (32–96: 0.281→0.039) and
  emptied 2 vehicle-heavy images. So v1 and v2 are **complementary**.
- **v3** (multi-scale: run both 640:1× and 512:2× per image, union + NMS): best overall —
  recall +42% over v1, F1 highest, and 16–32px recall (0.219) beats *both* single scales
  because the two passes catch different objects.

## Honest limits (won't yield to more tuning)

1. **<8px recall is a hard 0.000** across all versions — below the VLM's effective
   resolution after downsampling. This is a ceiling of the VLM-as-annotator approach, not
   a hyperparameter.
2. **Hallucinated grid boxes** persist (evenly-spaced rows/columns of identical boxes),
   capping precision (~0.41). The anti-hallucination prompt reduced but did not remove them.
3. **Recall–precision tradeoff:** every recall gain cost some precision (FP from edge
   duplicates + hallucinations).

## Full-val headline (109 images, 7971 core-class GT boxes)

The full-val run confirms the smoke trend at scale, and more strongly: **recall up
5.4× while precision holds flat.**

| metric | integral VLM | **tiled v3** | gain |
|---|---|---|---|
| overall precision | 0.538 | 0.544 | **flat** (no precision cost) |
| overall recall | 0.045 | **0.243** | **5.4×** |
| overall F1 | 0.082 | **0.336** | 4.1× |
| TP / FP / FN | 355 / 305 / 7616 | 1940 / 1626 / 6031 | +1585 TP |
| recall <8px | 0.004 | 0.015 | 3.8× |
| recall 8–16px | 0.005 | **0.100** | **20×** |
| recall 16–32px | 0.024 | **0.255** | **10.6×** |
| recall 32–96px | 0.092 | 0.391 | 4.3× |
| recall ≥96px | 0.324 | 0.476 | 1.5× |
| pedestrian recall | 0.008 | **0.300** | **37×** |
| vehicle recall | 0.085 | 0.265 | 3.1× |

**The headline observation:** the gain is *inversely proportional to object size* —
1.5× for ≥96px, but 20× for 8–16px. Tiling adds recall exactly where the integral
VLM collapsed (small objects), and **precision stays flat (0.538 → 0.544)** despite
5.4× more true positives, because the class whitelist + giant-box filter suppress the
scenery/mislabel noise that more detections would otherwise introduce. Run time:
5814s (~1.6h) for 109 images at 23 tiles/image.

### Two residual effects worth noting

1. **Tile-level truncation still happens.** On the densest tiles the per-tile JSON
   still overflows `max_new_tokens` (the salvage path fires, "salvaged N complete
   objects"). So the token ceiling is *fundamental*, not fully solved — finer tiles or
   higher `max_new_tokens` could push recall further.
2. **23 / 109 images returned zero detections** (GT∩VLM = 86). Their GT counts as pure
   FN, so the *effective* recall on non-empty images is higher than 0.243 — fixing the
   empty-image failure mode is the most promising next lever.

## Output

- `badcase.py` — GT-anchored P/R/F1 + per-size recall + ranked badcase image list.
- `tiled_vlm.py` — tiling inference; `--scales` multi-scale (default `640:1.0,512:2.0`),
  `--upscale`, `--max-box-frac`, class-constrained prompt + label whitelist.
- Smoke artifacts: `annotations_tiled_smoke{,_v2,_v3}.json` + badcase reports.

## Next

Run v3 on the full 109-image val for the headline integral-vs-tiled numbers, fill the
table above, and update the README with the finding. (Optional v4: an evenly-spaced-grid
filter to kill hallucinated rows/columns and recover precision — deferred as it risks
suppressing genuine rows of parked vehicles.)
