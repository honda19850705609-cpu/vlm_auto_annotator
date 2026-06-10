# Day 12 — Distillation: VLM pseudo-labels → train a detector → beat the teacher

**Date:** ~2026-06-10
**Track:** 方向一 × 方向二 (closes the loop)
**Goal:** The whole point of auto-annotation isn't the VLM's boxes — it's using them
to train a deployable detector *without human labels*. Test: can a detector trained
only on VLM pseudo-labels beat the VLM teacher, and how close does it get to a
detector trained on human labels?

## Setup

300 VisDrone-DET-**train** images (sampled across sequences). Three models, all
eval'd on the **val 109** with `badcase.py` (same metrics as the VLM):

1. **VLM teacher** — tiled v4 + de-hall (Day 11): F1 = 0.460.
2. **Detector / pseudo** — YOLOv8s trained on VLM pseudo-labels (zero human labels).
3. **Detector / real** — YOLOv8s trained on the 300 images' human GT (ceiling).

Pipeline: `tiled_vlm.py` (512:2) → `to_coco.py` → `coco_to_yolo.py` → `yolo detect
train` → `yolo_to_coco.py` → `badcase.py`. All on the 5090, ~8 min/training.

## The training labels are noisy

Pseudo-labels vs the 300 images' train GT: **P=0.499, R=0.410, F1=0.450**. So the
detector is learning from labels where ~half the boxes are wrong and ~60% of true
objects are missing.

## Three-way result (val 109, IoU=0.5)

| metric | VLM teacher | **detector / pseudo** | detector / real (ceiling) |
|---|---|---|---|
| F1 | 0.460 | **0.505** | **0.673** |
| precision | 0.509 | **0.764** | 0.715 |
| recall | 0.420 | 0.377 | **0.636** |
| speed | ≈100 s/img | ≈2 ms/img | ≈2 ms/img |

Recall by object size:

| size | VLM | pseudo-det | real-det |
|---|---|---|---|
| <8 | 0.075 | 0.044 | 0.228 |
| 8–16 | 0.187 | 0.141 | 0.446 |
| 16–32 | 0.404 | 0.346 | 0.666 |
| 32–96 | 0.692 | 0.672 | 0.845 |
| ≥96 | 0.859 | 0.768 | 0.876 |

## Findings

**1. The distilled detector beats the teacher.** F1 0.505 > 0.460, precision 0.764
≫ 0.509, at ≈10⁴× lower latency (≈100 s → ≈2 ms per image) — with **zero human
labels**.

**2. The student denoises the teacher.** It was trained on P=0.499 labels yet
reaches **P=0.764** on real GT — *cleaner than its own training labels*. A detector
learns the consistent signal (real objects) and discards the VLM's inconsistent,
non-generalizing false positives (grid hallucinations, edge duplicates).

**3. Zero human labels recovers ~75% of the supervised ceiling.**

| | pseudo / real |
|---|---|
| F1 | 0.505 / 0.673 = **75%** |
| precision | 0.764 / 0.715 = **107%** (exceeds) |
| recall | 0.377 / 0.636 = **59%** |

**4. The remaining gap is recall, concentrated in small objects.** <8px: 0.044 vs
0.228 (5×); 8–16px: 0.141 vs 0.446 (3×); but big objects nearly match (32–96px:
0.672 vs 0.845). Root cause: the VLM pseudo-labels *missed* those small objects in
the first place (training-label recall 0.41), so the detector never saw them — the
student's recall ceiling is inherited from the teacher's recall.

**5. Human labels still win clearly.** Even 300 human-labeled images give F1 0.673
vs the VLM's 0.460 — a trained detector substantially outperforms a zero-shot VLM.

## Bottom line

> **VLM auto-labels → train a detector is a viable zero-human-label pipeline:** it
> beats the VLM teacher (F1 +10%, precision +50%, ≈10⁴× faster) and recovers ~75%
> of the human-label F1 at 0% labeling cost. The 25% gap is small-object recall,
> bounded by the VLM's own recall — so the path to closing it is better small-object
> pseudo-label recall (finer tiles / super-resolution) or a little human labeling
> for small objects (semi-supervision).

This connects 方向二 (VLM annotation) to 方向一 (detector): the VLM is the
label engine, the detector is the deployable product.

## Output

- `coco_to_yolo.py` — COCO (pseudo/real) → YOLO training format, 3 core classes.
- `yolo_to_coco.py` — trained-YOLO predictions → COCO for `badcase.py`.
- `devlog/distill_runbook.md` — full 5090 pipeline.
- `results/distill/` — three-way reports + comparison.

## Next (optional)

- Sweep the detector's confidence (it *is* calibrated, unlike the VLM) to trade the
  spare precision for recall and push F1 past 0.505.
- Improve small-object pseudo-label recall, or add semi-supervision, to close the
  recall gap to the ceiling.
- Scale the train subset beyond 300 images.
