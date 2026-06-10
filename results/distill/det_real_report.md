# Badcase 报告 (IoU=0.5) — Detector trained on human GT (val 109, ceiling)

YOLOv8s trained on the same 300 VisDrone-train images' **human** GT, evaluated on
the val 109 vs human GT. This is the supervised upper bound for the 300-image budget.

### Detector (real / ceiling) vs GT
- 总体: P=0.715  R=0.636  F1=0.673  (TP=5069 FP=2023 FN=2902)
- 分类别:
    - pedestrian  P=0.629 R=0.501 F1=0.558 (TP=1356 FP=801 FN=1351)
    - vehicle     P=0.801 R=0.794 F1=0.798 (TP=3009 FP=747 FN=779)
    - bicycle     P=0.597 R=0.477 F1=0.530 (TP=704 FP=475 FN=772)
- 按目标尺寸分档的召回率:
    - <8     recall=0.228  (124/544)
    - 8-16   recall=0.446  (921/2065)
    - 16-32  recall=0.666  (1902/2857)
    - 32-96  recall=0.845  (1960/2320)
    - >=96   recall=0.876  (162/185)
