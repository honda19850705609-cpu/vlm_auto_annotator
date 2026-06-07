
# Day 2 — Forcing Structured JSON Output from the VLM
 
## Goal
Make the VLM emit detection-style structured JSON (label / bbox / confidence)
reliably, and parse it robustly. The real target is parse stability, not
detection accuracy.
 
## What I did
- Wrote `vlm_structured.py` with three layers of control over the output:
  1. A system prompt that fixes the role ("you only output a JSON array").
  2. A user prompt that gives an explicit schema plus one concrete example.
  3. A robust `extract_json` function with three fallbacks: direct parse,
     strip markdown code fences, then regex-grab the outer `[ ... ]` block.
- Added a `validate` step that checks each item has the required keys and a
  4-element bbox, and reports valid / malformed counts.
## Results & observations
- Aerial night-city image: the model returned an empty array `[]`. The parser
  handled it cleanly (0 parsed, 0 malformed) without crashing.
- Street-scene image: the model returned valid JSON but wrapped it in a
  ```json ... ``` markdown fence. A direct `json.loads` would have failed here;
  the fence-stripping fallback recovered it. Final result: 1 parsed, 1 valid,
  0 malformed, saved to output.json.
## Key insight
VLM output formatting is inherently unstable. With the *same* script and prompt,
one image produced a bare empty array and another produced fenced JSON.
Stability cannot be guaranteed by prompting alone; it must be enforced in the
parsing layer. This is exactly why a production annotation pipeline needs
fallback parsing rather than trusting the model to always comply.
 
Two boundaries of VLM detection also became clear:
- On dense aerial scenes it tends to give up and return nothing, because it
  knows it cannot produce precise pixel coordinates.
- Even on easy scenes its recall is low (one car detected in a busy street).
This confirms the positioning for the project: the VLM is an
understanding / quality-check layer, not a detector. It should judge or audit a
specialized detector's output (the Day 9 bad-case idea), not replace it.
 
## Next step (Day 3)
Switch tracks to the small-model side: export a lightweight detector (or
DINO-DETR) to ONNX, dealing with operator-compatibility and dynamic-shape
issues.