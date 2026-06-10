# Results — Distillation: VLM pseudo-labels → detector vs human-label ceiling

300 VisDrone-train images, three models eval'd on **val 109** vs human GT
(`badcase.py`, IoU=0.5). Full writeup: [`../../devlog/day12.md`](../../devlog/day12.md).

## Three-way comparison

| metric | VLM teacher (v4+dehall) | **detector / pseudo** (0 human labels) | detector / real (ceiling) |
|---|---|---|---|
| F1 | 0.460 | **0.505** | **0.673** |
| precision | 0.509 | **0.764** | 0.715 |
| recall | 0.420 | 0.377 | **0.636** |
| latency | ≈100 s/img | ≈2 ms/img | ≈2 ms/img |

Recall by object size:

| size | VLM | pseudo-det | real-det |
|---|---|---|---|
| <8 | 0.075 | 0.044 | 0.228 |
| 8–16 | 0.187 | 0.141 | 0.446 |
| 16–32 | 0.404 | 0.346 | 0.666 |
| 32–96 | 0.692 | 0.672 | 0.845 |
| ≥96 | 0.859 | 0.768 | 0.876 |

## Headline

- **The distilled detector beats its VLM teacher** — F1 0.505 > 0.460, precision
  0.764 ≫ 0.509, ≈10⁴× faster — trained on **zero human labels**.
- **It denoises the teacher**: trained on P=0.499 pseudo-labels, it reaches P=0.764
  on real GT (cleaner than its own training labels).
- **Zero human labels recovers ~75% of the supervised F1** (0.505 / 0.673), and
  *exceeds* its precision (107%).
- **The 25% gap is recall on small objects** (<8px: 0.044 vs 0.228), inherited from
  the VLM pseudo-labels that missed those objects to begin with.

## Files

- `det_pseudo_report.md` — detector trained on VLM pseudo-labels.
- `det_real_report.md` — detector trained on human GT (ceiling).
- Pseudo-label quality vs train GT (the training labels themselves): P=0.499,
  R=0.410, F1=0.450 — noisy, yet trainable.
