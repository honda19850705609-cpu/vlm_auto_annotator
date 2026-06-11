#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sr_preprocess.py —— 超分预处理层(叠在切图之上,攻 <8px 分辨率墙)

思路:VLM 切图标注的墙在"分辨率"——8px 目标降采样后看不清。在 VLM 看图**之前**,
用 Real-ESRGAN 把图放大 N×,给小目标"造"出学习到的细节(而非简单插值)。之后照常
跑 tiled_vlm(瓦片尺寸也×N),输出框在 SR 坐标系,再用 to_coco.py --box-scale 1/N
缩回原图坐标评估。这一层与切图正交,可叠加。

依赖(在 5090 上,加速已开):
  pip install realesrgan basicsr
  # 权重 RealESRGAN_x4plus.pth 首次自动下载(GitHub),加速开了应该能下;
  # 下不动就手动放到 --model-path 指向的位置。

退化方案:若 Real-ESRGAN 装不上,--method lanczos 用高质量插值(只是更好的放大,
无学习细节,作对照基线)。

用法:
  python sr_preprocess.py --in smoke5/ --out smoke5_sr2 --scale 2
  python sr_preprocess.py --in smoke5/ --out smoke5_lanczos2 --scale 2 --method lanczos
"""

import argparse
import os

from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_realesrgan(scale, model_path):
    """返回一个 upsample(np_bgr)->np_bgr 的函数;失败抛异常。"""
    import numpy as np  # noqa
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    # RealESRGAN_x4plus:x4 模型;想要 x2 就在 outscale 控制
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                    num_grow_ch=32, scale=4)
    up = RealESRGANer(scale=4, model_path=model_path, model=model,
                      tile=512, tile_pad=10, pre_pad=0, half=True)

    def run(img_bgr):
        out, _ = up.enhance(img_bgr, outscale=scale)
        return out
    return run


def main():
    ap = argparse.ArgumentParser(description="Super-resolution preprocessing before VLM tiling")
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=2.0, help="放大倍数(默认 2)")
    ap.add_argument("--method", choices=["realesrgan", "lanczos"], default="realesrgan")
    ap.add_argument("--model-path",
                    default="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                    help="Real-ESRGAN 权重路径或 URL")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = sorted(f for f in os.listdir(args.in_dir) if f.lower().endswith(IMG_EXTS))

    upsampler = None
    if args.method == "realesrgan":
        try:
            upsampler = load_realesrgan(args.scale, args.model_path)
            print(f">>> Real-ESRGAN loaded, scale x{args.scale}")
        except Exception as e:
            print(f"[WARN] Real-ESRGAN 不可用({type(e).__name__}: {e}); 回退 lanczos")
            args.method = "lanczos"

    n = 0
    for fn in files:
        src = os.path.join(args.in_dir, fn)
        img = Image.open(src).convert("RGB")
        W, H = img.size
        nW, nH = int(round(W * args.scale)), int(round(H * args.scale))
        if args.method == "realesrgan":
            import numpy as np
            bgr = np.array(img)[:, :, ::-1]            # RGB->BGR
            out = upsampler(bgr)[:, :, ::-1]           # BGR->RGB
            out_img = Image.fromarray(out).resize((nW, nH))
        else:
            out_img = img.resize((nW, nH), Image.LANCZOS)
        out_img.save(os.path.join(args.out, fn), quality=95)
        n += 1
        print(f"  [{n}/{len(files)}] {fn}: {W}x{H} -> {nW}x{nH}")

    print(f"[OK] {n} images -> {args.out}  (scale x{args.scale}, {args.method})")
    print(f"    next: tiled_vlm on {args.out} with x{args.scale} tiles, "
          f"then to_coco.py --box-scale {1/args.scale:.4f} --images <ORIGINAL imgs>")


if __name__ == "__main__":
    main()
