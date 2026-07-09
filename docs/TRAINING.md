# VideoNeuMat — training & advanced usage

The one-click **text / image → neural material** pipeline is in the main [README](../README.md). This document covers training the models, the cluster launchers, running the LRM on your own videos, and the dataset internals.

---

## Sample scripts

`sample_scripts/` has one ready-to-run command per task. Each is short and takes extra options on the command line. The main ones:

| Script | What it does |
| --- | --- |
| `sample_scripts/mlp_optimization_512mat.sh` | Pre-train the MLP on 512 materials (step 1). |
| `sample_scripts/cache_vae.sh` | Build the VAE latent cache (step 2). |
| `sample_scripts/LRM_optimization_all_dense.sh` | Train the LRM at 256 (step 3). |
| `sample_scripts/LRM_optimization_all_dense_hres.sh` | Train the LRM at 1024 (step 4). |
| `sample_scripts/inference_1024.sh` | Inference at 1024 on diffusion inputs. |
| `sample_scripts/inference_512.sh` | Inference at 512 on diffusion inputs. |
| `sample_scripts/mlp_optimization_1mat.sh` | Train the MLP on one material (for debugging). |

### `scripts/`

To run a job on the cluster, copy the script you want from `sample_scripts/` into `scripts/`, then give its name to a launcher (see below). `scripts/` is in `.gitignore`, so it is your own folder and starts empty after cloning.

The names used in the pipeline below are just copies you make once:

```bash
cp sample_scripts/mlp_optimization_512mat.sh         scripts/pretrain_mlp.sh       # step 1
cp sample_scripts/cache_vae.sh                        scripts/cache_vae.sh          # step 2
cp sample_scripts/LRM_optimization_all_dense.sh       scripts/train_lrm_lowres.sh   # step 3
cp sample_scripts/LRM_optimization_all_dense_hres.sh  scripts/train_lrm_hires.sh    # step 4
```

---

## Running on the cluster (SLURM)

Submit a script from `scripts/` through a launcher in `sample_scripts/cluster/`:

- `sample_scripts/cluster/multi_node_sig.sh` — runs any job on one or more nodes, saves a checkpoint on preemption and auto-requeues. Set the number of nodes with `--nodes` (use `--nodes=1` for a single node).
- `sample_scripts/cluster/interactive.sh` — grab an interactive GPU node inside the container.

Override the account / GPUs / nodes on the `sbatch` line (these win over the `#SBATCH` defaults inside the file):

```bash
sbatch --account nvr_rtr_<account> --gres=gpu:<X> --nodes=<Y> \
       sample_scripts/cluster/multi_node_sig.sh <script_name> [max_retries] [folder_name]
```

where:

- `<script_name>` is a file in `scripts/` **without the `.sh`** (e.g. `pretrain_mlp`).
- `max_retries`: `0` = no retries (default), `-1` = unlimited, `N` = at most N requeues.
- `folder_name`: output folder under `outputs/slurm/` (default: `$SLURM_JOB_ID`); reuse it to resume.

Logs land in `outputs/slurm/<job_id>/logs.log`.

---

## The full pipeline

> Steps 1 → 3 → 4 are the training chain; step 2 is a one-time preprocessing that the LRM (steps 3/4) depends on.

### 1. Pre-train the MLP

Train the shared neural-material decoder on many materials at low resolution:

```bash
sbatch --account <your_slurm_account> --gres=gpu:8 --nodes=4 \
       sample_scripts/cluster/multi_node_sig.sh pretrain_mlp 5
# or locally: bash sample_scripts/mlp_optimization_512mat.sh
```

This produces an `mlp.pth` checkpoint (under the run's output folder). Every LRM run is seeded from it via `checkpoint.network=`.

### 2. Build the VAE latent cache (for the LRM input datasets)

The LRM takes a few **representative images** of a material as input (`rep_images`) and reads them as pre-encoded **VAE latents** (`rep_latents`) — the `wan_lrm` latent uses `upsampler: vae` and `receives_vae_latent: True` by default, so this cache is required.

You only need the cache for datasets used as **input to the LRM** (the `rep_images`, i.e. the `sparse` views). Variants used only for the rendering loss do **not** need a cache.

Set the dataset in `src/config/cache_vae_latents.yaml` (or override it), then run:

```bash
sbatch --account <your_slurm_account> --gres=gpu:8 --nodes=1 \
       sample_scripts/cluster/multi_node_sig.sh cache_vae
```

The GPU/node count here is arbitrary — it's a quick preprocessing step. **But** note that `cache_vae` (like inference) cannot resume if it gets interrupted on the cluster, so pick a count that finishes within the walltime (8 or 16 GPUs is usually a good choice). After it finishes you must move the result into the dataset folder — see [VAE latent cache](#vae-latent-cache) for that and the sanity check.

### 3. Train the LRM at low resolution

Feed-forward LRM training at 256 on all materials, seeded from the MLP:

```bash
sbatch --account <your_slurm_account> --gres=gpu:8 --nodes=8 \
       sample_scripts/cluster/multi_node_sig.sh train_lrm_lowres -1
```

`sample_scripts/LRM_optimization_all_dense.sh` already wires `checkpoint.network=<mlp.pth>` (to one of my runs under `outputs/good/`). Point it at your own MLP:

```bash
bash sample_scripts/LRM_optimization_all_dense.sh checkpoint.network=/path/to/mlp.pth
```

This produces a `latent_module.pth` checkpoint (the LRM/encoder weights).

> During LRM training (steps 3 and 4) you can also make the MLP trainable, e.g. `trainable.network=true` (the high-res script does this). If you do, the run saves an **updated** `mlp.pth` — pass that updated MLP as `checkpoint.network` to the next step, not the original step-1 one.

### 4. Continue training the LRM at high resolution

Resume from the low-res LRM and train at 1024, using the checkpoints saved by step 3 (the updated `mlp.pth` if you trained the MLP there, plus the low-res `latent_module.pth`). `LRM_optimization_all_dense_hres.sh` already sets both checkpoint paths — edit them there (or in your `scripts/` wrapper) to point at your step-3 outputs:

```bash
bash sample_scripts/LRM_optimization_all_dense_hres.sh \
     checkpoint.network=/path/to/step3/mlp.pth \
     checkpoint.latent=/path/to/step3/latent_module.pth
```

Then submit it like step 3:

```bash
sbatch --account <your_slurm_account> --gres=gpu:8 --nodes=8 \
       sample_scripts/cluster/multi_node_sig.sh train_lrm_hires -1
```

---

## Inference

Inference lives here — two ready examples: `sample_scripts/inference_1024.sh` and `sample_scripts/inference_512.sh`. They are "training with no optimization": they already point at the known-good checkpoints (full paths in my folder, readable by everyone) and set `optimize=False`, so you can run them as-is. If you swap in your own checkpoints, make sure the architecture/material flags you trained with (e.g. `network`, `latent/upsampler`, `network.min_z`, `network.max_offset`, `use_offset`) match the checkpoint, otherwise the state dict won't line up.

Copy the example into `scripts/` and submit it on the cluster:

```bash
cp sample_scripts/inference_1024.sh scripts/inference_1024.sh
sbatch --account <your_slurm_account> --gres=gpu:8 --nodes=1 \
       sample_scripts/cluster/multi_node_sig.sh inference_1024
```

(Same for `inference_512`.) These scripts feed **diffusion-generated** material captures to the LRM (e.g. the 1024 script uses `val_dataset=[diffusion_1024x1024_sparse_all]`), set `optimize=False`, `latent/upsampler=vae`, and the inference-time `network.min_z` / `network.max_offset`. Swap `val_dataset` / checkpoints for your own.

> Inference (like `cache_vae`) cannot resume if interrupted on the cluster, so pick a GPU count that finishes within the walltime — 8 or 16 GPUs is usually a good choice.

> **LRM caveat:** the LRM needs the [VAE latent cache](#vae-latent-cache) for whatever dataset you feed it as input (`rep_images`). If the cache is missing, precompute it first (step 2).

### Running inference on your own videos

To run on your own videos (one video per material, e.g. a turntable/varying-light capture), build a dataset that matches the layout the loader expects. Use `data/diffusion/test` as a working reference and mirror it.

**1. Create the folder structure** under `data/diffusion/<split>/sparse/`:

```
data/diffusion/<split>/sparse/
├── <material_1>/btf_sequence.mp4     # your video for this material
├── <material_2>/btf_sequence.mp4
├── ...
├── metadata_clean.csv                # lists the materials
└── 0_scene_state/                    # per-frame camera + light (btf_frame__XXXX.npy), trajectory-specific
```

- `<split>` must be a **single token with no underscores** (e.g. `myvideos`, `1024x1024`) — the dataset name is split on `_` into exactly four parts.
- Each material is a folder containing `btf_sequence.mp4`, your input **video** for that material (`input_type: video`). Its frames are the views the LRM reads as input.
- `metadata_clean.csv` has the header `video,prompt` and one row per material, where the material name is the first path component:

  ```csv
  video,prompt
  <material_1>/btf_sequence.mp4,""
  <material_2>/btf_sequence.mp4,""
  ```

- `0_scene_state/` holds the per-frame **camera and light** info (`btf_frame__0000.npy`, …), and it is **trajectory-specific** — each trajectory (`sparse`, `dense`, `grazing`, …) has its own cameras and lights (`data/diffusion/<split>/<traj>/0_scene_state`). Copy it from the **same trajectory** of an existing dataset whose rig matches, e.g. for a `sparse` capture: `cp -r data/diffusion/test/sparse/0_scene_state data/diffusion/<split>/sparse/`. For a `grazing` capture, copy that trajectory's `0_scene_state` instead (and use `grazing` as the `<traj>` in the dataset name).

**2. Config.** You can reuse the existing `diffusion` base config (`src/config/datasets/diffusion.yaml`) as-is — it already resolves `data/diffusion/{split}/{traj}`, so a new `<split>` folder works with the name `diffusion_<split>_sparse_all`. Only add/edit a `src/config/datasets/*.yaml` if you need a different base name or different frame settings (`frames_info` / `rep_frame_info`).

**3. Build the VAE latent cache** for it (the LRM input), then move it into the dataset's `0 latents/` (see [VAE latent cache](#vae-latent-cache)):

```bash
bash sample_scripts/cache_vae.sh dataset=diffusion_<split>_sparse_all resolution=1024
# then move <out>/latents/<frames>-1024/ → data/diffusion/<split>/sparse/0 latents/<frames>-1024/
```

**4. Run inference** pointing `val_dataset` at your dataset (copy the example into `scripts/` and submit it):

```bash
cp sample_scripts/inference_1024.sh scripts/inference_mine.sh
# edit scripts/inference_mine.sh: val_dataset=[diffusion_<split>_sparse_all]
sbatch --account <your_slurm_account> --gres=gpu:8 --nodes=1 \
       sample_scripts/cluster/multi_node_sig.sh inference_mine
```

---

## VAE latent cache

The LRM reads its input `rep_images` as pre-encoded VAE latents (`rep_latents`). Generate them with `sample_scripts/cache_vae.sh` (which runs `src/cache_vae_latents.py`). Only the datasets fed to the LRM as input need this — datasets used only for the rendering loss do not.

### How to run it

1. **Set the dataset in the YAML.** The dataset is read from `src/config/cache_vae_latents.yaml` (the `dataset:` field), *not* from `train_dataset`. Edit it or override on the command line:

   ```bash
   bash sample_scripts/cache_vae.sh dataset=matsynth8_train_sparse_all resolution=1024
   ```

   Only one dataset at a time is supported.

2. **Run it** (locally as above, or on the cluster via `scripts/cache_vae.sh` + a launcher).

3. **Sanity check.** With `sanity_check: true` (default), it decodes a few random materials right after encoding and writes `<orig>` vs `<decoded>` side-by-side PNGs into a `sanity_check/` subfolder of the output. Eyeball these before trusting the latents.

4. **Move the result into the dataset folder.** The script writes latents to its run output, e.g.:

   ```
   <out_root>/latents/<sorted_frame_list>-<resolution>/<material>.pt
   ```

   but the data loader expects them under the dataset's representative-images path:

   ```
   data/<dataset>/<split>/sparse/0 latents/<sorted_frame_list>-<resolution>/<material>.pt
   ```

   The script prints the exact destination ("After generation, copy to data folder: ..."). Copy/move the generated `<sorted_frame_list>-<resolution>` folder into the dataset's `0 latents/` directory. **Do not rename it** — the folder name (sorted frame list + resolution) is how the loader finds it.

Once the cache is in place it is a **one-time cost**: the LRM loads the `.pt` latents directly on every run (no VAE re-encoding), and you never need to run `cache_vae` again for that dataset — unless you change the input frames or resolution.

---

## Diffusion inputs

Besides rendered datasets, the repo ships **diffusion-generated** input datasets (`data/diffusion/...`, configs `src/config/datasets/diffusion*.yaml`, `alldiffusion.yaml`, `gtcomparison.yaml`). These are material captures synthesized by a Wan video model (`src/models/diffsynth_wan_dit/`) and used as `rep_images` fed to the LRM — mostly for inference / qualitative comparison against rendered ground truth. Naming follows the same scheme as everything else (e.g. `diffusion_i2v_sparse_all`, `diffusion_1024x1024_sparse_all`).

---

## Dataset naming format

Datasets are referenced as strings parsed in `src/data.py::build_configs`:

```
<name>_<split>_<traj>_<materialset>
```

Example: `matsynth8_train_dense_512` → name `matsynth8`, split `train`, traj `dense`, materialset `512`.

- **`<name>`** — the base dataset config, `src/config/datasets/<name>.yaml` (e.g. `matsynth8`, `diffusion`, `gtcomparison`).
- **`<split>`** — substituted into the dataset paths (`{split}`), e.g. `train` / `test` (datasets live under `data/<name>/<split>/...`).
- **`<traj>`** — camera view sampling, substituted as `{traj}`. Usually **`sparse`** (the LRM input views) or **`dense`** (more views, for the rendering loss).
- **`<materialset>`** — a material preset under `src/config/datasets/presets/materials/<materialset>.yaml` (e.g. `512`, `10`, `hardest10`, `512first10`, `leather`), **or** the literal `all` to use every material in the split.

It must be exactly **four** underscore-separated parts (this is asserted), so don't put underscores inside any single field.

---

## Tips

- Flat-plane materials use a height/displacement, so the offset flags matter: `use_offset` (default `True`), `network.min_z`, `network.max_offset`. Keep them consistent between training and inference.
- `dataset.frames_info.every_other` controls frame subsampling (e.g. `8` → every 8th frame).
- Hydra builds the run name (`exp_name`) from key settings, so output folders are self-describing.

---
