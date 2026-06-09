# Results — VLM auto-annotation quality vs VisDrone GT

Full VisDrone-DET val subset: **109 images, 7971 core-class GT boxes**
(pedestrian / vehicle / bicycle). Matching = per-image-per-class greedy IoU≥0.5,
aligned by file name. Produced by `badcase.py`.

## Headline: tiling recovers recall 5.4× while precision holds flat

| metric | integral VLM | **tiled v3** | gain |
|---|---|---|---|
| precision | 0.538 | 0.544 | **flat** (no precision cost) |
| recall | 0.045 | **0.243** | **5.4×** |
| F1 | 0.082 | **0.336** | 4.1× |
| TP / FP / FN | 355 / 305 / 7616 | 1940 / 1626 / 6031 | +1585 TP |

### Recall by object pixel size — the gain is inversely proportional to size

| size bin | integral | tiled v3 | gain |
|---|---|---|---|
| <8px | 0.004 | 0.015 | 3.8× |
| 8–16px | 0.005 | **0.100** | **20×** |
| 16–32px | 0.024 | **0.255** | **10.6×** |
| 32–96px | 0.092 | 0.391 | 4.3× |
| ≥96px | 0.324 | 0.476 | 1.5× |

Integral (whole-image) VLM annotation collapses on small objects (the token
ceiling + `smart_resize` downsampling make sub-32px objects nearly invisible).
**Tiling** — overlapping tiles, multi-scale (640px 1× for medium + 512px 2×
upscaled for tiny), boxes mapped back to global coords with cross-tile NMS —
adds recall exactly where the integral model failed, without hurting precision
(class whitelist + giant-box filter suppress the scenery/mislabel noise).

## Folders

- `integral_baseline/` — whole-image `structured_vlm.py` output vs GT.
- `tiled_v3/` — `tiled_vlm.py --scales 640:1.0,512:2.0` output vs GT.

Each holds `report.md` (human-readable P/R/F1 + per-size recall), `result.json`
(machine-readable), and `badcase_images.txt` (worst images, ranked).

See `../devlog/day9.md` for the full method, the v1→v2→v3 ablation, and the
honest limits (<8px hard floor, residual grid hallucinations, 23/109 empty
images).
