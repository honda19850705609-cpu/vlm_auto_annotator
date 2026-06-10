# Distillation runbook — VLM pseudo-labels → train a detector → beat the teacher

**Hypothesis:** a detector trained *only* on VLM pseudo-labels (zero human labels)
can exceed the VLM teacher itself, because the detector sees full-resolution
images and learns VisDrone object appearance. Three-way compare, all eval'd on
the **val 109** with `badcase.py` (same metrics as VLM):

1. **VLM teacher** — v4 + de-hall: R=0.420 (already measured).
2. **Detector / pseudo-labels** — the distillation result.
3. **Detector / real labels** *(optional ceiling)* — supervised upper bound.

Paths assume the Day10 repo at `/root/autodl-tmp/Day10/vlm_auto_annotator` and
work dir `/root/autodl-tmp/distill`.

## 0. Prereqs (on 5090)

```bash
pip install ultralytics
mkdir -p /root/autodl-tmp/distill && cd /root/autodl-tmp/distill
```
Put **~300 VisDrone-DET-train images** in `train300/` and (optional, for the
ceiling arm) their COCO GT at `train300_gt.json`.

## 1. VLM pseudo-label the 300 train images (the slow step)

Single-scale 512:2 keeps it ~4h (multi-scale would be ~9h); the detector tolerates
slightly noisier labels.

```bash
cd /root/autodl-tmp/Day10/vlm_auto_annotator
PYTHONUNBUFFERED=1 nohup python -u tiled_vlm.py \
  --model /root/autodl-tmp/qwen2.5-vl-7b \
  --image-dir /root/autodl-tmp/distill/train300 \
  --out /root/autodl-tmp/distill/pseudo.json \
  --scales 512:2.0 \
  > /root/autodl-tmp/distill/pseudo.log 2>&1 &
tail -f /root/autodl-tmp/distill/pseudo.log     # wait for TILED BATCH DONE

python to_coco.py --in /root/autodl-tmp/distill/pseudo.json \
  --images /root/autodl-tmp/distill/train300 \
  --out /root/autodl-tmp/distill/pseudo_coco.json
```

## 2. Build YOLO datasets (3 core classes)

```bash
cd /root/autodl-tmp/Day10/vlm_auto_annotator
# pseudo-label dataset
python coco_to_yolo.py --coco /root/autodl-tmp/distill/pseudo_coco.json \
  --images /root/autodl-tmp/distill/train300 \
  --out /root/autodl-tmp/distill/ds_pseudo --split train
# (optional) real-label dataset for the ceiling
python coco_to_yolo.py --coco /root/autodl-tmp/distill/train300_gt.json \
  --images /root/autodl-tmp/distill/train300 \
  --out /root/autodl-tmp/distill/ds_real --split train
```

## 3. Train YOLOv8s (small + fast; aerial → train at imgsz 1280)

```bash
yolo detect train data=/root/autodl-tmp/distill/ds_pseudo/data.yaml \
  model=yolov8s.pt imgsz=1280 epochs=80 batch=8 \
  project=/root/autodl-tmp/distill/runs name=pseudo
# optional ceiling
yolo detect train data=/root/autodl-tmp/distill/ds_real/data.yaml \
  model=yolov8s.pt imgsz=1280 epochs=80 batch=8 \
  project=/root/autodl-tmp/distill/runs name=real
```

## 4. Predict on val 109 → COCO

```bash
cd /root/autodl-tmp/Day10/vlm_auto_annotator
python yolo_to_coco.py \
  --weights /root/autodl-tmp/distill/runs/pseudo/weights/best.pt \
  --images /root/autodl-tmp/visdrone_val_gt109 \
  --out /root/autodl-tmp/distill/det_pseudo_coco.json --imgsz 1280 --conf 0.25
# optional ceiling
python yolo_to_coco.py \
  --weights /root/autodl-tmp/distill/runs/real/weights/best.pt \
  --images /root/autodl-tmp/visdrone_val_gt109 \
  --out /root/autodl-tmp/distill/det_real_coco.json --imgsz 1280 --conf 0.25
```

## 5. Evaluate vs val GT — the comparison

```bash
GT=/root/autodl-tmp/Day10/vlm_auto_annotator/visdrone_val_gt.json   # or the val instances json
python badcase.py --gt $GT --vlm /root/autodl-tmp/distill/det_pseudo_coco.json \
  --out /root/autodl-tmp/distill/bc_det_pseudo
python badcase.py --gt $GT --vlm /root/autodl-tmp/distill/det_real_coco.json \
  --out /root/autodl-tmp/distill/bc_det_real      # optional
cat /root/autodl-tmp/distill/bc_det_pseudo/report.md
```

Then compare R/P/F1 + per-size recall against the VLM teacher (0.420). Send the
two `report.md`s back for the write-up.

## Notes

- `--conf 0.25` on the detector is a starting point; sweep it later for the best
  P/R point (the detector's own confidence *is* calibrated, unlike the VLM's).
- If the pseudo detector underperforms, likely causes: too few train images,
  noisy boxes (raise `--scales` quality), or imgsz too low for small objects.
