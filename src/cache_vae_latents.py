import torch
import hydra
import einops
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from omegaconf import DictConfig, OmegaConf
from torch.utils.data._utils.collate import default_collate
from accelerate import Accelerator
from PIL import Image
import sys
sys.path.append("./")

from src.models.wan_lrm import load_LRM_vae
from src.data import load_datasets, build_configs
from src.utils import to_pil
from accelerate.logging import get_logger

logger = get_logger(__name__)

def encode_vae(vae, image_seq: torch.Tensor, min_value=-1, max_value=1, device=None):
    """Encode images using VAE."""
    assert torch.all(image_seq.min() >= 0) and torch.all(image_seq.max() <= 1.0)
    assert image_seq.ndim == 5
    image_seq = einops.rearrange(image_seq, 'b t h w c -> b c t h w')
    image_seq = image_seq * (max_value - min_value) + min_value
    return vae.encode(image_seq, device=device)




def decode_vae(vae, latents: torch.Tensor, device=None):
    """Decode latents using VAE. Returns images in [0, 1] range."""
    if latents.ndim == 4:
        latents = latents.unsqueeze(0)
    decoded = 0.5 * (1 + vae.decode(latents, device=device))
    decoded = einops.rearrange(decoded, 'b c t h w -> b t h w c')
    return decoded.clamp(0, 1)


def sanity_check(vae, latent_dir, output_dir, cache, device):
    """Load saved latents, decode, and compare with cached originals."""
    output_dir = output_dir / "sanity_check"
    os.makedirs(output_dir, exist_ok=True)
    
    for name, orig_images in cache.items():
        logger.info(f"Sanity checking {name}")
        latent = torch.load(latent_dir / f"{name}.pt")
        decoded = decode_vae(vae, latent, device=device)[0]
        orig_row = np.hstack([np.array(to_pil(orig_images[t])) for t in range(len(orig_images))])
        dec_row = np.hstack([np.array(to_pil(decoded[t])) for t in range(len(decoded))])
        Image.fromarray(orig_row).save(output_dir / f"{name}_orig.png")
        Image.fromarray(dec_row).save(output_dir / f"{name}_dec.png")




@hydra.main(version_base=None, config_path="config", config_name="cache_vae_latents")
def main(cfg: DictConfig) -> None:
    output_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device
    torch.set_grad_enabled(False) 
    
    # Load VAE
    vae = load_LRM_vae(Path(cfg.vae_path), True, False)
    vae.eval()
    vae.requires_grad_(False)
    
    # Build configs from dataset string, modify frames_info to use rep_frame_info
    assert isinstance(cfg.dataset, str), "Only one dataset is supported, more is not tested"
    cfgs = build_configs([cfg.dataset])
    
    
    for name, _cfg in cfgs.items():
        _cfg.frames_info = _cfg.rep_frame_info
        _cfg.rep_frame_info = None
        _cfg.materials_info.exclude = []
    
    dataset = load_datasets(
        tonemapping_type='none',
        unrolled=False,
        resolution=cfg.resolution,
        disable_rep=True,
        rep_use_latent=None,
        cfgs=cfgs,  # pass pre-built configs
        build_cfg=False,
    )
    
    logger.info(f"Dataset: {dataset.summary()}")
    # Create dataloader

    frames = dataset.datasets[0].frame_list
    n_frames = len(frames)

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers, 
        shuffle=False,
    )

    vae, dataloader = accelerator.prepare(vae,dataloader)

    # Get rep_images_path from the first dataset config to determine the expected structure
    # The data loader expects latents at: {rep_images_path}/0 latents/{sorted(frames)}-{resolution}/{material_name}.pt
    # build_configs already replaces {split} and {traj} in the config, so rep_images_path is already formatted
    first_cfg = list(cfgs.values())[0]
    rep_images_path = Path(first_cfg.rep_images_path)
    
    # Construct the output directory where latents should be saved (in output folder, not data folder)
    # User typically copies these to the data folder after generation
    sorted_frames_str = str(sorted(frames))
    out_dir = output_path / 'latents' / f'{sorted_frames_str}-{cfg.resolution}'
    os.makedirs(out_dir, exist_ok=True)
    
    # Also log where they should be copied to
    expected_data_dir = rep_images_path / '0 latents' / f'{sorted_frames_str}-{cfg.resolution}'
    logger.info(f"Saving latents to output folder: {out_dir}")
    logger.info(f"After generation, copy to data folder: {expected_data_dir}")
    
    # Cache for sanity check - we'll just take the first N materials we process
    sanity_check_cache = {}  # name -> original images
    
    logger.info(f"Starting encoding loop: {len(dataloader)} batches, output_dir={out_dir}")
    
    pbar = tqdm(total=len(dataloader), disable=not accelerator.is_main_process, dynamic_ncols=False, mininterval=10)
    executor = ThreadPoolExecutor(max_workers=cfg.num_workers)
    save_futures = []

    for batch_idx, batch in enumerate(dataloader):
        images = batch['images']
        names = batch['names']
        ids = batch['mat_ids']
        bs = len(ids)
        
        # Skip batch if all files already exist
        all_exist = all(os.path.exists(f"{out_dir}/{name}.pt") for name in names)
        if all_exist:
            logger.info(f"Skipping batch {batch_idx}/{len(dataloader)}, names {names}, ids {ids.tolist()}, images {images.shape} because all files already exist")
            pbar.update(1)
            continue
        
        logger.info(f"Processing batch {batch_idx}/{len(dataloader)}, names {names}, ids {ids.tolist()}, images {images.shape}")
        
        # Cache first N materials for sanity check BEFORE encoding (while images are on GPU)
        for i in range(bs):
            if len(sanity_check_cache) < cfg.sanity_check_samples:
                sanity_check_cache[names[i]] = images[i].cpu()
        
        # Encode with VAE
        encoded_images = encode_vae(vae, images, device=device).cpu()
        
        # Clear VAE's internal feature cache and GPU memory
        unwrapped_vae = accelerator.unwrap_model(vae)
        if hasattr(unwrapped_vae, 'clear_cache'):
            unwrapped_vae.clear_cache()
        del images
        torch.cuda.empty_cache()
        
        # Submit async saves (non-blocking)
        for i in range(bs):
            future = executor.submit(torch.save, encoded_images[i].clone(), f"{out_dir}/{names[i]}.pt")
            save_futures.append(future)
        
        # Clear completed futures to avoid memory buildup
        save_futures = [f for f in save_futures if not f.done()]
        
        pbar.set_postfix(names=names, ids=ids.tolist(), encoded=encoded_images.shape)
        del encoded_images
        pbar.update(1)

    # Wait for all async saves to complete
    for future in save_futures:
        future.result()
    executor.shutdown()

    # Wait for all GPUs to finish saving before sanity check
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        logger.info(f"Sanity checking {len(sanity_check_cache)} materials")
        sanity_check(accelerator.unwrap_model(vae), out_dir, output_path, sanity_check_cache, device)


if __name__ == "__main__":
    main()
