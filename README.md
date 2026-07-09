# VideoNeuMat: Neural Material Extraction from Generative Video Models

**SIGGRAPH 2026**

Bowen Xue, Saeed Hadadan, Zheng Zeng, Fabrice Rousselle, Zahra Montazeri, Milos Hasan

**[Project Page](https://bowenxueai.github.io/VideoNeuMat/) · [Paper](https://github.com/bowenxueai/VideoNeuMat/releases/download/full-resolution/VideoNeuMat_SIG_2026_full_resolution.pdf) · [Supplementary](https://bowenxueai.github.io/VideoNeuMat/assets/supplementary_web.pdf) · [Video](https://bowenxueai.github.io/VideoNeuMat/assets/neumat_overview.mp4)**

Official code for **VideoNeuMat**. The main pipeline is **text → neural material**: a text prompt is turned into a material video by a fine-tuned video-diffusion model, then a feed-forward LRM extracts a re-renderable neural material (BRDF + displacement) from it — in one command (see **Quick start** below).

> The codebase is internally named **NeuGen**; scripts, configs, and module paths use that name.

---

NeuGen learns neural **materials** on a flat plane from rendered data, in two flavors that are trained as a pipeline:

- **MLP (per-material optimization)** — a shared neural-material decoder optimized directly on rendered data.
- **LRM (feed-forward)** — a Large Reconstruction Model that *encodes* a material into a latent in one forward pass, then decodes it with the pre-trained MLP.

The LRM input is a few **representative images** of a material (`rep_images`); it reads them as pre-encoded **VAE latents** (`rep_latents`). The repo also includes **diffusion-generated** input datasets (e.g. `diffusion`, made with a Wan video model under `src/models/diffsynth_wan_dit/`) that can be fed to the LRM in place of rendered captures — this is what the inference scripts use.

---

## Quick start — one command: text / image → neural material

`release/oneclick_material.sh` runs the whole pipeline: a fine-tuned **Wan2.1** video
model turns a prompt (or an image) into an 81-frame 1024² material video, then the
**LRM** extracts a re-renderable neural material (+ novel-view / reconstruction renders).

```bash
# text -> material  (T2V full generator, the default)
release/oneclick_material.sh "The video clip features a close-up view of <material>, <micro-geometry details>"

# text -> material  (lightweight LoRA generator)
release/oneclick_material.sh --lora "<prompt>"

# image -> material  (I2V generator, from a still image)
release/oneclick_material.sh --i2v material.jpg "<prompt>"
```

The commands above submit two chained SLURM jobs. **No cluster?** Run the exact same pipeline
on one local GPU with **`release/run_local.sh`** (same flags) — it needs two Python envs built
from `init/` (`GEN_ENV` for generation, `NEUGEN_ENV` for the LRM) instead of the container.

Prerequisites, weights (HuggingFace [`yley/videoneumat`](https://huggingface.co/yley/videoneumat)),
and a one-shot `release/download_weights.sh` are documented in **[release/README.md](release/README.md)**.

---

## Setup

Clone the repo:

```bash
git clone https://github.com/bowenxueai/videoneumatcode.git
cd videoneumatcode
```

### Requirements

Pip requirements and a conda environment live in `init/` — pick whichever you prefer:

- `init/requirements.txt` / `init/conda_requirements.txt`
- `init/environment.yml`

If you hit installation issues, install torch first, then install the rest of the env.

### Docker / container (recommended for the cluster)

A ready-made image is the easiest path and is what the cluster scripts use.

- The image is built from `init/Dockerfile` (base `nvcr.io/nvidia/pytorch:25.09-py3` + the `neugen` conda env).
- On the cluster we run a prebuilt enroot squashfs at `$DISK/cache/neugen.sqsh`.
- The conda env inside the container is `neugen`; activate it with `source /opt/miniconda/bin/activate neugen`.

> The cluster launchers use a `$DISK` environment variable to find the code and the image (`$DISK/neugen` and `$DISK/cache/neugen.sqsh`). It is **not** set by default — define it yourself to point at the folder that holds the checkout and `cache/`, e.g. `export DISK=$HOME/videoneumat`. Add it to your shell profile so it is always set.

### Data

The code reads model weights and datasets from a relative `data/` folder (the Wan VAE/DiT under `data/models/`, rendered/diffusion datasets under `data/matsynth8/...` and `data/diffusion/...`).

- **Model weights + pretrained checkpoints** are released on HuggingFace — run `release/download_weights.sh` (see [release/README.md](release/README.md)); it fetches the Wan base, this project's fine-tuned material DiT, and the LRM checkpoints, and prints where to point `data/models/` and the checkpoint paths.
- **Rendered training datasets** (`data/matsynth8`, …) were rendered internally and are **not** part of this release; they can be (re)created with `src/generate_data.py` (see `sample_scripts/datagen.sh`). The one-click **text → material** pipeline (see **Quick start** (above)) does **not** need them.

### Pretrained checkpoints & results

The released checkpoints are on HuggingFace **[`yley/videoneumat`](https://huggingface.co/yley/videoneumat)** (fetched by `release/download_weights.sh`):

- **LRM** — `lrm/latent_module.pth`: encodes a material into a latent in one feed-forward pass.
- **Universal MLP** — `lrm/mlp.pth`: the shared neural-material decoder (kept training through the high-res stage, so it is the better universal MLP — prefer it).
- **Material generators** (Stage A — each turns a prompt/image into an 81-frame 1024² material video that then feeds the LRM):
  - `wan14b/step-10000.safetensors` — **T2V, full** fine-tuned Wan2.1-T2V-14B DiT. The default; run with `GEN_MODE=full`.
  - `wan14b_t2v_lora/step-9000.safetensors` — **T2V, LoRA** on base Wan2.1-T2V-14B (293 MB). Run with `GEN_MODE=lora MATERIAL_CKPT=.../step-9000.safetensors`.
  - `wan14b_i2v/step-9500.safetensors` — **I2V, full** fine-tuned Wan2.1-I2V-14B DiT (image→video). Needs the `Wan-AI/Wan2.1-I2V-14B` base + an input image (not wired into the one-click, which is T2V; use it with an i2v `WanVideoPipeline`).

All three produce the same 81-pose sparse-rig material video the LRM consumes, so any of them can front the pipeline.

---

## Training & advanced usage

Training the MLP/LRM, the SLURM cluster launchers, running the LRM on your own videos, the VAE-latent cache, and the dataset format are documented in **[docs/TRAINING.md](docs/TRAINING.md)**.

---

## Citation

```bibtex
@inproceedings{xue2026videoneumat,
  author    = {Xue, Bowen and Hadadan, Saeed and Zeng, Zheng and Rousselle, Fabrice and Montazeri, Zahra and Hasan, Milos},
  title     = {VideoNeuMat: Neural Material Extraction from Generative Video Models},
  booktitle = {ACM SIGGRAPH 2026 Conference Papers},
  year      = {2026},
}
```
