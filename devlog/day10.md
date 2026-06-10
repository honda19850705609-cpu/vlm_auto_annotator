# Day 10 — Refining the VLM pipeline: confidence analysis + de-hallucination

**Date:** ~2026-06-10
**Track:** 方向二 (VLM data analysis)
**Goal:** Push the tiled-VLM result (Day 9: R=0.243, P=0.544) further along both
axes — precision and recall — and understand which knobs actually move the needle.
All measurements reuse `badcase.py` (file-name aligned, core-class normalized,
greedy IoU≥0.5) on the full 109-image val.

## (1) Confidence thresholding is a weak lever — a useful negative result

`analyze_confidence.py` sweeps the VLM's self-reported `confidence` and re-scores
at each threshold.

| thr | P | R | F1 |
|---|---|---|---|
| 0.0–0.6 | 0.544 | 0.243 | 0.337 |
| 0.8 | 0.568 | 0.223 | 0.320 |
| 0.85 | 0.650 | 0.158 | 0.254 |
| 0.95 | 1.000 | 0.001 | 0.001 |

Finding: **VLM confidence is poorly calibrated for this task.** Scores cluster in
the high band, so P/R are flat from thr 0→0.6; the only way to lift precision is
to also crush recall (0.65 precision costs 35% of recall). Best F1 sits at
thr≈0.6 (i.e. ~no filtering). **Precision must be bought structurally, not by
thresholding confidence.** (PR curve: `results/refinements/pr_curve.png`.)

## (2) De-hallucination filter — the right precision lever

The negative result above motivated attacking FPs by *structure*. The dominant
structured-FP pattern is the **grid hallucination**: when asked to enumerate many
small objects, the model emits a run of same-label boxes that are *pixel-identical
in size* and *collinear with constant spacing* (real objects jitter; these don't).

`dehallucinate.py` flags, per (image, class), same-size collinear clusters of
≥`min_run` boxes (one center axis std < `line_tol`) and drops them. Criteria are
deliberately conservative (pixel-level identical size + near-zero jitter) to spare
genuine rows of parked vehicles.

| | tiled v3 | + de-hallucination |
|---|---|---|
| precision | 0.544 | **0.579** (+3.5pt) |
| recall | 0.243 | 0.243 (flat) |
| F1 | 0.336 | **0.342** |
| FP | 1626 | **1411** (−215) |

Dropped **215 FP across 16 grid clusters at the cost of 2 TP** — a clean precision
win, per-size recall unchanged. Honest caveat: 1411/1626 FPs remain — grid
hallucinations are only ~13% of all FPs; the rest are mislocalized boxes and edge
duplicates, which this filter does not address.

## Output

- `analyze_confidence.py` — threshold sweep → PR curve + best operating point.
- `dehallucinate.py` — COCO-in/COCO-out grid-hallucination filter (`--min-run`,
  `--size-tol`, `--line-tol`); chain before `badcase.py`.
- `results/refinements/` — PR curve, pr_table.json, de-hallucinated report.

## Next

- **Empty-image recovery** (biggest remaining recall lever): 23/109 images return
  zero detections and hold **930 GT boxes = 11.5% of all GT**, currently all FN.
  Whole sequences go empty (e.g. `0000242_*` ×6) → looks like a scene-type cause
  (night / low-light / high-altitude tiny-only). Plan: diagnose on sample images,
  add an empty-image fallback (finer tiles / higher upscale / relaxed prompt).
- **Open-vocab detector baseline** (context): run Grounding DINO / YOLO-World on
  the same 109 to position VLM auto-annotation against a real open-vocab detector.
