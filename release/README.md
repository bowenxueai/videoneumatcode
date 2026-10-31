# VideoNeuMat — one-click: text prompt → neural material

Generate a material video from a text prompt with a fine-tuned **Wan2.1-T2V-14B**
model, then extract a **NeuGen neural material** (a re-renderable BRDF+displacement)
from that video with the LRM — in a single command.

```
prompt ──[diffsynth · Wan2.1-T2V-14B]──▶ 81-frame 1024² material video
       ──[LRM]──▶ neural material  +  novel-view / reconstruction renders
```

## Files

All code is in the repo — **no external repo to clone**. The `diffsynth` generation library is
vendored at the repo root (`./diffsynth`, Apache-2.0); the LRM's DiT/VAE subset is `src/models/diffsynth_wan_dit/`.

| file | role |
|---|---|
| `oneclick_material.sh` | entrypoint — submits the two chained SLURM jobs |
| `gen_video.sbatch` | **Stage A**: Wan2.1-T2V-14B text→video (vendored `diffsynth`) |
| `gen_material_video.py` | the generation script (vendored from `DiffSynth-Studio/train_neugen/infer_prompts.py`) |
| `extract_material.sbatch` | **Stage B**: build dataset → cache VAE latents → LRM inference (NeuGen container) |
| `config.env` | all external paths (edit this after obtaining the weights) |
| `README.md` | this file |

## Usage

```bash
release/oneclick_material.sh "The video clip features a close-up view of weathered basalt cobblestones, sharp angular facets, moss in the crevices"
# batch:
release/oneclick_material.sh --prompts_csv my_prompts.csv     # CSV with a 'prompt' column
```

It prints two job ids (gen → extract) and where results land:
```
outputs/release_runs/<run>/videos/0000.mp4                                   # generated video
outputs/release_runs/<run>/infer/final/<run>_split/sparse_traj/mat000/
    val_views.mp4     # material rendered on a 320-frame novel-view turntable  ← main result
    train_views.mp4   # reconstruction of the input trajectory
    latents/*.exr     # the neural-material latent + displacement
```

**Prompt convention** (the material fine-tune's trigger phrase):
`"The video clip features a close-up view of <material>, <micro-geometry / relief / color details>"`,
empty negative prompt. Cost ≈ **70 min/video on one 80 GB GPU** (50 diffusion steps) + a few minutes for extraction.

## Prerequisites — the heavy assets (NOT in git; ~150 GB)

Nothing large is committed. Obtain these once and point `config.env` at them:

1. **Generation Python env** — the `diffsynth` library is **vendored in this repo** (`./diffsynth`),
   so there is no external repo to clone. Just make a Python env with the generation deps:
   `pip install -r init/gen_requirements.txt` (install a torch build matching your CUDA first). Set `GEN_ENV` to it.
2. **Wan2.1-T2V-14B base weights** (~65 GB) — public on HuggingFace:
   ```bash
   huggingface-cli download Wan-AI/Wan2.1-T2V-14B  --local-dir $WAN_MODELS/Wan-AI/Wan2.1-T2V-14B
   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B "google/*" --local-dir $WAN_MODELS/Wan-AI/Wan2.1-T2V-1.3B
   ```
   Set `WAN_MODELS` to that root (`DIFFSYNTH_MODEL_BASE_PATH`).
3. **Material generator** — HuggingFace [`yley/videoneumat`](https://huggingface.co/yley/videoneumat). Three variants, all feed the LRM; set `MATERIAL_CKPT` (and `GEN_MODE`):
   - `t2v/videoneumat-t2v.safetensors` — **T2V full** (default, `GEN_MODE=full`, 27 GB)
   - `t2v_lora/videoneumat-t2v-lora.safetensors` — **T2V LoRA** (`GEN_MODE=lora`, 293 MB, needs the Wan2.1-T2V-14B base)
   - `i2v/videoneumat-i2v.safetensors` — **I2V full** (image→video; requires the Wan2.1-I2V-14B-720P base and an input image). The pipeline defaults to T2V; select I2V with `--i2v material.jpg "<prompt>"` in either `oneclick_material.sh` or `run_local.sh`.
4. **NeuGen LRM checkpoints** — same repo `yley/videoneumat`: `lrm/videoneumat-lrm.pth` (6 GB) + `lrm/videoneumat-decoder.pth`. Set `LRM_LATENT`, `LRM_MLP`.
5. **NeuGen container** — the `neugen` enroot squashfs (build from the repo's `init/Dockerfile`, or import the prebuilt image). Set `NEUGEN_CONTAINER`.

> **Shortcut:** `release/download_weights.sh [DEST]` fetches items 2–4 (this project's weights + the Wan base + tokenizer) and prints the exact `config.env` values to set.

> **Publishing to GitHub:** commit only `release/` (code) + the rest of the NeuGen source.
> Keep every `.safetensors` / `.pth` / `.sqsh` out of git (they exceed GitHub's 100 MB limit).
> Recommended: put `videoneumat-t2v.safetensors` and the two LRM checkpoints in a HuggingFace model
> repo and link them here; the Wan base is already publicly downloadable and `diffsynth` is vendored in-repo.
> A `download_weights.sh` that wraps the `huggingface-cli` commands above is the friendliest touch.

## License

The original VideoNeuMat code and the LRM checkpoints (`lrm/videoneumat-lrm.pth` and `lrm/videoneumat-decoder.pth`) are licensed under the **Creative Commons Attribution NonCommercial 4.0 International (CC BY-NC 4.0) license**. The material-generator checkpoints in `t2v/`, `t2v_lora/`, and `i2v/` retain **Apache-2.0**, the same license as the upstream Wan2.1 models. Third-party code and separately downloaded base weights retain their original licenses. See the [repository license section](../README.md#license) for the full scope and exceptions.

## How it works

- **Stage A** (`gen_video.sbatch`) runs `gen_material_video.py --mode full --ckpt $MATERIAL_CKPT`
  in the generation env (`GEN_ENV`) using the vendored `./diffsynth`: it loads the base Wan2.1-T2V-14B, overwrites the DiT with the
  fine-tuned checkpoint, and renders `81×1024×1024, 50 steps, cfg 5.0, empty negative` →
  `save_video(fps=15, yuv444p)`. One array task per prompt.
- **Stage B** (`extract_material.sbatch`, `--dependency=afterok`) runs inside the NeuGen container:
  it lays the generated mp4(s) out as a `data/diffusion/<run>/sparse` dataset (81 frames map 1:1
  to the LRM's sparse BTF rig, camera/light in `shared/scene_state_sparse`), caches the VAE latents
  (`src/cache_vae_latents.py`), then runs the feed-forward LRM (`src/inverse_render_LRM.py optimize=False`)
  to encode the material and render it.

Every tunable (resolution, steps, cfg, seed, account, partition, all paths) is in `config.env`
and can also be overridden per-invocation via the environment.
