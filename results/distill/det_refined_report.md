# Badcase 报告 (IoU=0.5) — D1: SAHI self-trained detector (val 109)

YOLOv8s self-trained one round: D0 (pseudo) re-labels the 300 train images via
SAHI tiled inference, fused with the VLM pseudo-labels → refined labels → train D1
(warm-start from D0). Zero human labels. Evaluated on val 109 vs human GT.

### Detector D1 (self-train) vs GT
- 总体: P=0.608  R=0.477  F1=0.535  (TP=3804 FP=2457 FN=4167)
- 分类别:
    - pedestrian  P=0.585 R=0.355 F1=0.442 (TP=961 FP=683 FN=1746)
    - vehicle     P=0.643 R=0.667 F1=0.655 (TP=2528 FP=1404 FN=1260)
    - bicycle     P=0.460 R=0.213 F1=0.292 (TP=315 FP=370 FN=1161)
- 按目标尺寸分档的召回率:
    - <8     recall=0.103  (56/544)
    - 8-16   recall=0.258  (532/2065)
    - 16-32  recall=0.473  (1350/2857)
    - 32-96  recall=0.740  (1717/2320)
    - >=96   recall=0.805  (149/185)

vs D0 (pseudo): F1 0.505→0.535, recall 0.377→0.477 (+27%), <8px 0.044→0.103 (2.3×),
8-16px 0.141→0.258 (1.8×). Ceiling (real) F1=0.673 → recovery 75%→79.5%.
