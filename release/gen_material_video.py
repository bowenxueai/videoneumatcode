#!/usr/bin/env python
"""Batch text-to-video inference for Wan2.1-T2V-14B over a prompt CSV.

Sharded inference: each worker renders prompts[rank::world_size]. Sharding source:
  - SLURM job array  -> rank=SLURM_ARRAY_TASK_ID-MIN, world=SLURM_ARRAY_TASK_COUNT (1 GPU/task)
  - else srun ntasks -> rank=SLURM_PROCID, world=SLURM_NTASKS
Each prompt -> one mp4. Existing outputs are skipped (resumable).

Modes: base | full (pipe.dit.load_state_dict) | lora (pipe.load_lora).
--low_vram enables CPU-offload VRAM management (for 1024x1024 on a single GPU).
"""
import argparse, csv, os, torch
from diffsynth.utils.data import save_video
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

NEG = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
       "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
       "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
NEG_REALISM = ("Screenshot from a video game, scene from a video game, digital image, computer-generated image, "
               "3D rendering, digitally created, graphic, 3D rendered image, graphic design, in-game shot, CGI-rendered scene")
POS_REALISM = ", Photograph, high-resolution, 4K, high-quality."


def read_prompts(path):
    prompts = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        col = "prompt" if "prompt" in (r.fieldnames or []) else (r.fieldnames or ["prompt"])[0]
        for row in r:
            p = (row.get(col) or "").strip()
            if p:
                prompts.append(p)
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["base", "full", "lora"], required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--prompts_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg_scale", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora_alpha", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--quality", type=int, default=10, help="imageio video quality 0-10 (10=highest)")
    ap.add_argument("--pix_fmt", type=str, default="yuv444p", help="ffmpeg pixel format: yuv444p (no chroma subsampling) / yuv420p")
    ap.add_argument("--lossless", action="store_true", help="near-lossless H.264 (-crf 0) with the chosen pix_fmt")
    ap.add_argument("--low_vram", action="store_true", help="CPU-offload VRAM management for big res on one GPU")
    ap.add_argument("--neg_style", default=os.environ.get("NEG_STYLE", "wan"), help="negative prompt: wan | realism")
    ap.add_argument("--pos_style", default=os.environ.get("POS_STYLE", "none"), help="positive suffix: none | realism")
    args = ap.parse_args()

    # --- sharding (job-array aware) ---
    if os.environ.get("SLURM_ARRAY_TASK_COUNT"):
        world_size = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        amin = int(os.environ.get("SLURM_ARRAY_TASK_MIN", "0"))
        global_rank = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) - amin
    else:
        world_size = int(os.environ.get("SLURM_NTASKS", "1"))
        global_rank = int(os.environ.get("SLURM_PROCID", "0"))
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    ndev = max(1, torch.cuda.device_count())
    dev_idx = local_rank % ndev
    torch.cuda.set_device(dev_idx)
    device = f"cuda:{dev_idx}"
    print(f"[rank {global_rank}/{world_size}] device={device} mode={args.mode} res={args.height}x{args.width} low_vram={args.low_vram}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    prompts = read_prompts(args.prompts_csv)
    if global_rank == 0:
        with open(os.path.join(args.out_dir, "prompts_index.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["index", "prompt"])
            for i, p in enumerate(prompts):
                w.writerow([i, p])
    if os.environ.get("ABS_INDEX") == "1":
        _ai = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        mine = [(_ai, prompts[_ai])] if 0 <= _ai < len(prompts) else []
    else:
        mine = [(i, p) for i, p in enumerate(prompts) if i % world_size == global_rank]
    todo = [(i, p) for i, p in mine if not os.path.exists(os.path.join(args.out_dir, f"{i:04d}.mp4"))]
    print(f"[rank {global_rank}] {len(todo)}/{len(mine)} to generate ({len(mine)-len(todo)} already exist)", flush=True)
    if not todo:
        print(f"[rank {global_rank}] nothing to do; exiting before model load", flush=True); return

    cfg_extra, vram_kwargs = {}, {}
    if args.low_vram:
        cfg_extra = dict(offload_dtype=torch.bfloat16, offload_device="cpu",
                         onload_dtype=torch.bfloat16, onload_device="cpu",
                         preparing_dtype=torch.bfloat16, preparing_device="cuda",
                         computation_dtype=torch.bfloat16, computation_device="cuda")
        total = torch.cuda.mem_get_info(dev_idx)[1] / (1024 ** 3)
        vram_kwargs["vram_limit"] = total - 4
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors", **cfg_extra),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **cfg_extra),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-14B", origin_file_pattern="Wan2.1_VAE.pth", **cfg_extra),
        ],
        redirect_common_files=False,
        **vram_kwargs,
    )
    if args.mode == "full":
        assert args.ckpt, "--ckpt required for full"
        pipe.dit.load_state_dict(load_state_dict(args.ckpt))
    elif args.mode == "lora":
        assert args.ckpt, "--ckpt required for lora"
        pipe.load_lora(pipe.dit, args.ckpt, alpha=args.lora_alpha)
    print(f"[rank {global_rank}] model ready ({args.mode}); generating {len(todo)} (neg={args.neg_style} pos={args.pos_style} cfg={args.cfg_scale})", flush=True)
    neg_used = NEG_REALISM if args.neg_style == "realism" else ("" if args.neg_style == "none" else NEG)
    pos_used = POS_REALISM if args.pos_style == "realism" else ""
    for i, prompt in todo:
        out = os.path.join(args.out_dir, f"{i:04d}.mp4")
        if os.path.exists(out):
            continue
        video = pipe(
            prompt=prompt + pos_used, negative_prompt=neg_used,
            height=args.height, width=args.width, num_frames=args.num_frames,
            cfg_scale=args.cfg_scale, num_inference_steps=args.steps,
            seed=args.seed, tiled=True,
        )
        ff = ["-pix_fmt", args.pix_fmt]
        if args.lossless:
            ff += ["-crf", "0", "-preset", "veryslow"]
        save_video(video, out, fps=args.fps, quality=args.quality, ffmpeg_params=ff)
        print(f"[rank {global_rank}] wrote {out} (q={args.quality} pix_fmt={args.pix_fmt} lossless={args.lossless})", flush=True)
    print(f"[rank {global_rank}] DONE", flush=True)


if __name__ == "__main__":
    main()
