"""Minimal VLM single-image inference (Day 1).

Loads a Qwen2.5-VL model, runs inference on one image, and prints the output
together with basic latency / VRAM statistics.

Usage:
    python minimal_vlm.py \
        --model /root/autodl-tmp/qwen2.5-vl-7b \
        --image /root/autodl-tmp/test.jpg \
        --prompt "Describe this image and list every object you can see."
"""

import argparse
import time

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

DEFAULT_PROMPT = (
    "Describe this image, and list every distinct object you can see "
    "with its rough location."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal VLM single-image inference.")
    parser.add_argument(
        "--model",
        default="/root/autodl-tmp/qwen2.5-vl-7b",
        help="Path to a local model dir, or a HuggingFace repo id.",
    )
    parser.add_argument(
        "--image",
        default="/root/autodl-tmp/test.jpg",
        help="Path to the input image.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Text instruction sent to the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f">>> loading model from: {args.model}")
    t0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model)
    print(f">>> model loaded in {time.time() - t0:.1f}s")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    print(">>> generating ...")
    t1 = time.time()
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    gen_time = time.time() - t1

    trimmed = [
        out[len(inp):] for inp, out in zip(inputs.input_ids, generated)
    ]
    output_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    n_new = trimmed[0].shape[0]
    print("\n===== MODEL OUTPUT =====\n")
    print(output_text)

    print("\n===== STATS =====")
    print(f"generation time : {gen_time:.1f}s")
    print(f"tokens generated: {n_new}")
    print(f"throughput      : {n_new / gen_time:.1f} tok/s")
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"peak VRAM       : {peak:.1f} GB")


if __name__ == "__main__":
    main()
