# Day 4 — VLM batch structured inference

## What I did
Combined the Day-2 single-image JSON logic with a folder-level batch loop into
one script (`structured_vlm.py`). The model loads once and is reused across all
images. Each image is wrapped in `try/except`, so a single failure (unreadable
or corrupt file) is logged and skipped rather than killing the batch. Output is
one aggregated `annotations.json` with per-image detections plus run-level
stats: `num_ok`, `num_failed`, `total_detections`, `elapsed_sec`.

## Bugs hit & fixed
1. **Fresh AutoDL instance retained nothing.** Switched machines; the new
   instance had no model weights, no deps, and no Cursor server. Re-installed
   `modelscope` / `transformers` / `qwen-vl-utils` and re-downloaded the 3B and
   7B weights. (Also: the SSH failure was a local proxy hijacking port 7897 plus
   a wrong region domain `westd` vs `westc` in the SSH config.)
2. **3B returned `[]` on a dense aerial scene** — insufficient grounding.
   Switched to 7B, which detected objects correctly. 7B is the practical
   minimum here.
3. **JSON output truncated by `max_new_tokens`** on dense images, raising
   `JSONDecodeError`. Hardened `extract_json` to salvage every complete `{...}`
   object and drop the trailing partial one; raised `max_new_tokens` to 4096.

## Key data
3 images, 3 ok / 0 failed, 8 total detections, 8.5 s. Bbox rescaling
(Qwen `smart_resize` space -> original pixels) verified visually on a
540x960 aerial street scene — boxes sit accurately on the targets. Confirmed
the rescaling also holds on a high-resolution image (bbox coords ~1300+),
so it is not specific to one resolution.

## Observation (D9 hook)
The 7B VLM recalls far fewer objects than a dedicated dense detector on aerial
street scenes (~5 boxes vs. dozens of real targets). This is the gap that
motivates using a VLM for **auto-labeling / bad-case triage** rather than as a
detector replacement. The empty-result image (`img3`) is itself a natural
bad-case candidate — a concrete starting point for the Day-9 bad-case picker.
