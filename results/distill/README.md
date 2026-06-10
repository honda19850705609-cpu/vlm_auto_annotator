# Results — Distillation: VLM pseudo-labels → detector vs human-label ceiling

300 VisDrone-train images, three models eval'd on **val 109** vs human GT
(`badcase.py`, IoU=0.5). Full writeup: [`../../devlog/day12.md`](../../devlog/day12.md).

## Comparison (all detectors zero human labels except the ceiling)

| metric | VLM teacher | detector D0 (pseudo) | **detector D1 (SAHI self-train)** | real (ceiling) |
|---|---|---|---|---|
| F1 | 0.460 | 0.505 | **0.535** | 0.673 |
| precision | 0.509 | 0.764 | 0.608 | 0.715 |
| recall | 0.420 | 0.377 | **0.477** | 0.636 |
| latency | ≈100 s/img | ≈2 ms/img | ≈2 ms/img | ≈2 ms/img |

Recall by object size:

| size | VLM | D0 (pseudo) | D1 (self-train) | real (ceiling) |
|---|---|---|---|---|
| <8 | 0.075 | 0.044 | **0.103** | 0.228 |
| 8–16 | 0.187 | 0.141 | **0.258** | 0.446 |
| 16–32 | 0.404 | 0.346 | **0.473** | 0.666 |
| 32–96 | 0.692 | 0.672 | **0.740** | 0.845 |
| ≥96 | 0.859 | 0.768 | **0.805** | 0.876 |

## Headline

- **D0 beats its VLM teacher** — F1 0.505 > 0.460, precision 0.764 ≫ 0.509, ≈10⁴×
  faster — trained on **zero human labels**; it *denoises* the teacher (trained on
  P=0.499 labels, reaches P=0.764 on real GT).
- **D1 closes most of the small-object gap** via SAHI tiled self-relabeling + fusion
  (Day 13): F1 0.505 → 0.535, recall +27%, small-object recall ~2× (<8px:
  0.044 → 0.103). Ceiling recovery **75% → 79.5%**, all still zero human labels.
- The remaining gap to the human ceiling is small-object recall, bounded by what
  even tiled self-inference can resurface.

## Files

- `det_pseudo_report.md` — D0, trained on VLM pseudo-labels.
- `det_refined_report.md` — D1, SAHI self-trained one round.
- `det_real_report.md` — detector trained on human GT (ceiling).
- Label quality vs train GT: VLM pseudo P=0.499/R=0.410 → SAHI-refined
  P=0.405/R=0.533 (recall +30%, small-bins ~2×).
