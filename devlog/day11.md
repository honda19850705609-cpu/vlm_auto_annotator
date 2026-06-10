# Day 11 — Empty-image bug: one prompt fix → recall 0.243 → 0.420

**Date:** ~2026-06-10
**Track:** 方向二 (VLM data analysis)
**Goal:** Chase the biggest remaining recall lever flagged on Day 9 — **23/109
val images returned zero detections**, holding 930 GT boxes (11.5% of all GT)
that were all counted as misses.

## Diagnosis: it was a prompt bug, not a model limit

The empty images turned out to be ordinary **daytime, object-dense** scenes — a
general VLM finds plenty on them — so "empty" had to be a pipeline bug. They
shared one trait: **low resolution** (e.g. 540×960, whole sequences like
`0000242_*`).

`diag_tile.py` isolated prompt vs tiling by running the *same* image two ways:

| input | prompt | result |
|---|---|---|
| whole 540×960 | `structured_vlm`'s generic `JSON_PROMPT` | **26 cars** |
| whole 540×960 | tiling's `VISDRONE_PROMPT` | **`[]`** (0) |
| each 640px tile | `VISDRONE_PROMPT` | **`[]`** (0) |

So the model *can* see the objects; `VISDRONE_PROMPT` made it answer `[]`. The
old prompt led with *"Detect EVERY individual **small** object"* + a stack of
*"Do NOT…"* + *"if you see no such object, respond with []"*. On low-res frames
where the cars don't look "small", the model took the offered exit and returned
`[]`. The constraints meant to curb scenery/hallucination were **over-suppressing
real detections**.

## Fix: recall-forward prompt; keep guardrails downstream

Rewrote `VISDRONE_PROMPT` to lead with *"Detect every distinct object … be
exhaustive"* (like the prompt that worked), demote the class list to guidance,
and keep only a light one-box-per-object / no-grid clause. The real guardrails
are **structural, not prompt-based**: `normalize_label` (whitelist) drops scenery
labels; `dehallucinate.py` drops grid hallucinations. Verified on the same frame:
`[]` → **33 / 26 / 17** detections (whole / tile0 / tile1).

## Full-val result (109 images, 7971 core GT) — v4

Empty images: **23 → 0**. Every image now has predictions.

| metric | integral | v3 (Day 9) | **v4 (empty-fix)** | **v4 + de-hall** |
|---|---|---|---|---|
| recall | 0.045 | 0.243 | **0.420** | **0.420** |
| precision | 0.538 | 0.544 | 0.439 | **0.509** |
| F1 | 0.082 | 0.336 | 0.430 | **0.460** |
| TP / FP | 355 / 305 | 1940 / 1626 | 3350 / 4276 | 3347 / 3230 |

Recall by size (v4 + de-hall):

| size | <8 | 8–16 | 16–32 | 32–96 | ≥96 |
|---|---|---|---|---|---|
| recall | 0.075 | 0.187 | 0.404 | 0.692 | 0.859 |

**The single prompt fix lifted recall 0.243 → 0.420 (+73%)** — the largest jump of
any change, and it touched every size bin (32–96px: 0.391 → 0.692). Precision
dipped (more recovered images + a bolder prompt = more FP, including more grid
hallucinations: de-hall now drops 1049 boxes / 50 clusters, vs 217 on v3).
**De-hallucination buys precision back 0.439 → 0.509 at zero recall cost.**

## Bottom line (best config = v4 + de-hallucination)

> **recall 0.420 (9.3× the integral baseline), precision 0.509, F1 0.460 (5.6×).**
> Precision is back near the baseline's 0.538 while recall is an order of magnitude
> higher.

## Lesson

Over-constraining a detection prompt can silently zero out whole image classes
(here: low-res frames). Push recall in the prompt; enforce precision with
*inspectable, measurable* post-filters (whitelist, de-hallucination) rather than
negative prompt instructions you cannot audit.

## Output

- `tiled_vlm.py` — recall-forward `VISDRONE_PROMPT`.
- `diag_tile.py` — prompt-vs-tiling isolation harness.
- `results/v4_emptyfix/`, `results/v4_emptyfix_dehall/` — full reports.

## Next (optional)

- Remaining FPs (3230) are mostly mislocalized / edge-duplicate boxes, not grids —
  a localization-aware NMS or edge-aware merge could lift precision further.
- Open-vocab detector baseline (Grounding DINO / YOLO-World) for context.
