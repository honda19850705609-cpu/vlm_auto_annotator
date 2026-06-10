# Day 13 — SAHI tiled self-training: closing the small-object gap, zero human labels

**Date:** ~2026-06-11
**Track:** 方向一 × 方向二
**Goal:** Day 12 left the distilled detector (D0) at F1 0.505 — 75% of the human
ceiling (0.673). The gap was **entirely small-object recall** (<8px: 0.044 vs
0.228). Close it without any human labels.

## Diagnosis: missing labels are *negative* supervision

The deeper reason D0's small-object recall is low isn't "it never saw them" — it's
worse. When the VLM pseudo-labels **miss** a small car, that pixel region is
unlabeled, so training treats it as **background**. The detector is therefore
*actively trained to suppress* small objects, not merely left ignorant of them.
The student's recall ceiling is inherited from the teacher's recall, and reinforced
by this missing-label penalty.

So the fix must **complete the labels** — find the missed small objects — using a
labeler better at small objects than the VLM. We can build one.

## Method: SAHI tiled self-relabeling + fusion (`sahi_relabel.py`)

A whole-image detector is no better than the VLM at <8px. But run that same
detector with **tiled inference (SAHI — Slicing Aided Hyper Inference)**: slice
each train image into 640px tiles at imgsz 1280, so small objects appear ~2× larger
and become detectable. Then **fuse** the detector's tiled detections with the VLM
pseudo-labels (union + per-class NMS) → more-complete labels → retrain.

```
D0 (pseudo-trained)
  → SAHI-relabel train300 with D0  ∪  VLM pseudo-labels   → refined labels
  → train D1 (warm-start from D0)  → re-evaluate on val
```

### The labels got more complete (vs train GT)

| label set | R | <8 | 8–16 | 16–32 | P |
|---|---|---|---|---|---|
| VLM pseudo (D0's labels) | 0.410 | 0.052 | 0.182 | 0.367 | 0.499 |
| **SAHI-refined (D1's labels)** | **0.533** | **0.099** | **0.314** | **0.510** | 0.405 |

Label recall +30%; small-bin label recall ~doubled. Precision dropped (tiling adds
edge-duplicate FPs) — but for *training* labels, completeness matters more, and the
detector denoises FPs anyway (Day 12).

## Result: self-training round 1 (val 109)

| metric | VLM teacher | D0 (pseudo) | **D1 (self-train)** | ceiling (real) |
|---|---|---|---|---|
| F1 | 0.460 | 0.505 | **0.535** | 0.673 |
| recall | 0.420 | 0.377 | **0.477** | 0.636 |
| precision | 0.509 | 0.764 | 0.608 | 0.715 |
| <8px | 0.075 | 0.044 | **0.103** | 0.228 |
| 8–16px | 0.187 | 0.141 | **0.258** | 0.446 |
| 16–32px | 0.404 | 0.346 | **0.473** | 0.666 |
| 32–96px | 0.692 | 0.672 | **0.740** | 0.845 |
| ≥96px | 0.859 | 0.768 | **0.805** | 0.876 |

**One self-training round, zero human labels:**
- F1 0.505 → 0.535; recall 0.377 → **0.477 (+27%)**.
- Small-object recall ~doubled: <8px **2.3×** (0.044→0.103), 8–16px **1.8×**.
- Ceiling recovery: F1 **75% → 79.5%**, recall **59% → 75%** — the recall gap that
  was the whole problem is now mostly closed.
- Precision dipped (0.764 → 0.608) as the refined labels were noisier; the
  detector's calibrated confidence is a knob to trade some back.

## Trajectory (all zero human labels until the ceiling)

```
VLM teacher (zero-shot)   F1 0.460
 → distilled detector D0  F1 0.505   (beats teacher, 75% of ceiling)
 → SAHI self-train D1     F1 0.535   (79.5% of ceiling, small-obj recall ~2×)
 → human ceiling          F1 0.673
```

## Takeaway

> The small-object gap in pseudo-label distillation is a **missing-label /
> negative-supervision** problem, not a model-capacity one. Completing the labels
> with **tiled self-inference (SAHI) + fusion** — using the detector's own
> resolution advantage that the VLM lacked — recovers it: +27% recall, small-object
> recall doubled, F1 to ~80% of the human ceiling, **without a single human label.**

## Output

- `sahi_relabel.py` — SAHI tiled relabeling of a trained detector + fusion with VLM
  pseudo-labels → completed COCO labels.
- `results/distill/det_refined_report.md` — D1 on val.

## Next (optional)

- **Round 2**: SAHI-relabel with D1 → train D2 (diminishing but likely +3–5%).
- Sweep D1's confidence to recover the spare precision.
- Add a small human-labeled set for <8px (semi-supervision) to chase the last gap.
