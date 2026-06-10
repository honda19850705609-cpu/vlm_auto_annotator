# Badcase 报告 (IoU=0.5) — Detector trained on VLM pseudo-labels (val 109)

YOLOv8s trained on 300 VisDrone-train images labeled by the VLM (zero human labels),
evaluated on the val 109 vs human GT.

### Detector (pseudo) vs GT
- 总体: P=0.764  R=0.377  F1=0.505  (TP=3006 FP=928 FN=4965)
- 分类别:
    - pedestrian  P=0.732 R=0.268 F1=0.393 (TP=726 FP=266 FN=1981)
    - vehicle     P=0.799 R=0.543 F1=0.647 (TP=2057 FP=516 FN=1731)
    - bicycle     P=0.604 R=0.151 F1=0.242 (TP=223 FP=146 FN=1253)
- 按目标尺寸分档的召回率:
    - <8     recall=0.044  (24/544)
    - 8-16   recall=0.141  (292/2065)
    - 16-32  recall=0.346  (988/2857)
    - 32-96  recall=0.672  (1560/2320)
    - >=96   recall=0.768  (142/185)
