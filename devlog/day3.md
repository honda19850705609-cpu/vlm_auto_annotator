# Day 3 — DINO-DETR: Environment, Inference Verification & ONNX Export
 
## Goal
Switch to the small-model track. Take a trained VisDrone DINO-DETR checkpoint
all the way to a working ONNX file: set up the environment, verify PyTorch
inference, then export and validate ONNX.
 
## Model under test
- DINO 5-scale, ResNet-50 backbone, custom modifications (BiFPN neck, P2 high-res
  level, channel attention, SIoU loss, VisDrone class weighting).
- num_classes=10 (VisDrone), num_queries=1200, hidden_dim=256.
- Config recovered directly from the checkpoint's stored `args`:
  DINO_5scale_visdrone_lightweight_100ep.py, trained to epoch 97 (EMA weights).
## Environment setup
- CUDA 12.8 shipped under /usr/local/cuda-12.8 but nvcc wasn't on PATH; fixed by
  exporting PATH/LD_LIBRARY_PATH and persisting in .bashrc.
- Installed DINO deps (timm, pycocotools, scipy, addict, yapf) plus onnx /
  onnxruntime-gpu.
## Hard problem 1: deformable-attention op (ABI break)
- The custom CUDA op (MultiScaleDeformableAttention) compiled, but importing it
  failed with `undefined symbol: c10_cuda_check_implementation`.
- Root cause: that C10 symbol's signature changed in torch 2.8; the op was built
  against an older PyTorch ABI. Clean rebuilds didn't help. Downgrading to torch
  2.5 fixed the ABI but then the toolkit rejected the RTX 5090 (Blackwell sm_120
  needs CUDA 12.8+) — the two requirements conflict.
- Fix: bypass the compiled op entirely. The repo ships a pure-PyTorch
  implementation (`ms_deform_attn_core_pytorch`, built on F.grid_sample). Wrapped
  the CUDA-op import in try/except and forced the attention module to always use
  the pure-PyTorch path. This is also exactly what ONNX export needs, since ONNX
  cannot consume custom CUDA ops.
## Hard problem 2: corrupt weight transfer
- The first checkpoint transfer (which hung for hours) truncated the file to
  157M; `zipfile.is_zipfile` reported CORRUPTED. gdown couldn't reach Google
  Drive from the instance (network unreachable). Re-uploaded the full 444M file
  via AutoDL's web upload; verified `valid`.
- Loading hit torch 2.6+'s new default `weights_only=True`, which rejects the
  stored argparse.Namespace. Fixed with `weights_only=False` (trusted source).
## Inference verification
- Built the model with the recovered config, loaded the EMA weights, ran one
  forward pass on a VisDrone aerial image.
- Result: 219 detections above 0.3, top score 0.93, correct class coverage
  (pedestrian / car / van / truck / tricycle / motor / bus / bicycle). The
  bypass path works correctly in real inference — model is ready for export.
## ONNX export
- Wrapped the model to take a plain (B,3,H,W) tensor and return pred_logits and
  pred_boxes only (postprocessing kept outside the graph). Exported with opset
  17 (needed for grid_sample), dynamic H/W axes.
- First attempt segfaulted at the graph-writing stage (GPU memory pressure +
  constant folding on the large 1200-query / 5-scale graph). Fixed by exporting
  on CPU with `do_constant_folding=False`.
- Validation: `onnx.checker` OK; ONNXRuntime runs and returns the correct shapes
  (pred_logits (1,1200,10), pred_boxes (1,1200,4)) with sane logit range
  (-11.3 to -2.2). ONNX file: 150M.
## Outcome
PyTorch -> ONNX export complete and validated. First item of the deployment
pipeline (PyTorch -> ONNX -> TensorRT) is done through ONNX.
 
## Takeaways
- On a brand-new GPU, an old detector's custom CUDA op can be unfixable by
  version juggling (old ABI vs. new toolkit conflict). Bypassing it with the
  pure-PyTorch path is the cleaner route — and is required for ONNX anyway.
- Large DETR graphs can segfault during ONNX writing; exporting on CPU with
  constant folding off is a reliable fallback.
- Keep environment setup scripted on the data disk; the system disk kept getting
  reset and re-installing by hand cost hours.
## Next step (Day 5)
ONNX -> TensorRT: build a TensorRT engine from this ONNX, then measure the
before/after comparison table (AP/F1, latency ms, model size MB, VRAM).
 