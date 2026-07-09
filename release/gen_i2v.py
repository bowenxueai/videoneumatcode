#!/usr/bin/env python
"""Image-to-video material generation with the fine-tuned Wan2.1-I2V-14B DiT (step-9500).
A material still + a prompt -> an 81-frame 1024x1024 material video that feeds the LRM.
Models resolve under DIFFSYNTH_MODEL_BASE_PATH (Wan-AI/Wan2.1-I2V-14B-720P for the base,
Wan-AI/Wan2.1-T2V-1.3B for the umt5 tokenizer). Uses the vendored ./diffsynth via PYTHONPATH.
"""
import argparse, os, torch
from PIL import Image
from diffsynth.utils.data import save_video
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="fine-tuned I2V DiT (step-9500.safetensors)")
    ap.add_argument("--image", required=True, help="input material still image")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_id", default="Wan-AI/Wan2.1-I2V-14B-720P")
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg_scale", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--quality", type=int, default=10)
    ap.add_argument("--pix_fmt", default="yuv444p")
    args = ap.parse_args()

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16, device="cuda",
        model_configs=[
            ModelConfig(model_id=args.base_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id=args.base_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id=args.base_id, origin_file_pattern="Wan2.1_VAE.pth"),
            ModelConfig(model_id=args.base_id, origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        redirect_common_files=False,
    )
    # overwrite the base I2V DiT with the fine-tuned material one
    pipe.dit.load_state_dict(load_state_dict(args.ckpt))

    image = Image.open(args.image).convert("RGB").resize((args.width, args.height))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"[i2v] generating {args.width}x{args.height}x{args.num_frames} from {args.image}", flush=True)
    video = pipe(
        prompt=args.prompt, negative_prompt="", input_image=image,
        height=args.height, width=args.width, num_frames=args.num_frames,
        cfg_scale=args.cfg_scale, num_inference_steps=args.steps, seed=args.seed, tiled=True,
    )
    save_video(video, args.out, fps=args.fps, quality=args.quality, ffmpeg_params=["-pix_fmt", args.pix_fmt])
    print(f"[i2v] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
