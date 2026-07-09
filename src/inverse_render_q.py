import sys
from typing import Any
import hydra
import gc
import time

from utils import remove_module_prefix, save_state, load_state_if_any, log_cuda_mem, is_resumption
from signal_handling import should_terminate, save_and_requeue, mark_finished

sys.path.append("./")
import numpy as np
import torch

from src.neumip_nets import initalize_network, initalize_latents
import os
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from src.common import (
    seed_everything,
    tonemapping,
    calc_loss,
    validation,
    render_torch,
    load_scene_states,
    tv_loss,
    create_scheduler,
    init_lpips,
    lpips_loss,
)
import math
from src.data import load_datasets
from torch.utils.data import RandomSampler, DataLoader
from accelerate import Accelerator, DistributedType
from accelerate.data_loader import SeedableRandomSampler
from accelerate.logging import get_logger
from src.models.wan_lrm import load_LRM_vae
from pathlib import Path

logger = get_logger(__name__)

torch.cuda.set_per_process_memory_fraction(0.98, device=0)


@hydra.main(version_base=None, config_path="config", config_name="inverse_render")
def main(cfg: DictConfig) -> None:
    output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    accelerator = Accelerator(mixed_precision=cfg.mixed_precision, gradient_accumulation_steps=cfg.grad_acc_steps)
    
    device = accelerator.device
    logger.info(f"[{accelerator.process_index}] Using device: {device}", main_process_only=False)
    logger.info(f"{accelerator.state}")
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(f"Working directory : {os.getcwd()}")
    logger.info(f"Output directory  : {output_path}")

    if cfg.seed is not None:
        seed_everything(cfg.seed)

    # Initialize metrics dict (will be populated if optimizing)
    metrics = {}
    
    with accelerator.main_process_first():
        train_dataset = load_datasets(
            tonemapping_type=cfg.tonemapped,
            unrolled=True,
            resolution=cfg.resolution,
            disable_rep=cfg.disable_rep_images_loading,
            rep_use_latent=cfg.latent.receives_vae_latent,
            cfgs=cfg.train_dataset,
            rep_resolution=cfg.latent.resolution,
            load_textures=cfg.use_falcor,
            texture_data_root=cfg.texture_data_root
        )

        logger.info(f"Train dataset: {train_dataset.summary()}")

        val_dataset = load_datasets(
            tonemapping_type=cfg.tonemapped,
            unrolled=False,
            resolution=cfg.resolution,
            disable_rep=cfg.disable_rep_images_loading,
            rep_use_latent=cfg.latent.receives_vae_latent,
            cfgs=cfg.val_dataset,
            rep_resolution=cfg.latent.resolution,
        )

        logger.info(f"Validation dataset: {val_dataset.summary()}")

        val_states = load_scene_states('data/matsynth_scene_state', 320)

    logger.info(f"Rank {accelerator.process_index}, {accelerator.local_process_index}: Loading latent module", main_process_only=False)

    with accelerator.local_main_process_first():
        latent_module = initalize_latents(
            cfg.latent.name,
            {
                **cfg.latent,
                "use_offset": cfg.use_offset,
                "material_names": train_dataset.real_names,
                "device": device,
                "input_resolution": cfg.latent.resolution,
                "n_input_frames": len(train_dataset.datasets[0].representative_frame_list),
            },
        )
    
    logger.info(f"Rank {accelerator.process_index}, {accelerator.local_process_index}: Latent module loaded", main_process_only=False)
    neural_brdf = initalize_network(
        {
            "use_offset": cfg.use_offset,
            "latent_channels": cfg.latent.channels,
            "min_z": cfg.network.min_z,
            "max_offset": cfg.network.max_offset,
            "network_width": cfg.network.width,
            "network_depth": cfg.network.depth,
        }
    )
    logger.info(f"Rank {accelerator.process_index}, {accelerator.local_process_index}: MLP loaded", main_process_only=False)

    # Initialize LPIPS for validation metrics
    lpips_fn = init_lpips(cfg.lpips.net, device)
    logger.info(f"LPIPS initialized with {cfg.lpips.net} backbone")

    with accelerator.local_main_process_first():
        if cfg.checkpoint.network is not None:
            missed = neural_brdf.load_state_dict(remove_module_prefix(torch.load(cfg.checkpoint.network, map_location=device)), strict=False)
            logger.info(f"Network checkpoint loaded, missing: {missed}")
            del missed
            gc.collect()
        if cfg.checkpoint.latent is not None:
            logger.info(f"Loading a checkpoint from {cfg.checkpoint.latent}")
            if is_resumption(output_path, cfg.resume_path)[0] == 0:
                missed = latent_module.load_state_dict(remove_module_prefix(torch.load(cfg.checkpoint.latent, map_location=device)), strict=False)
                logger.info(f"Latent checkpoint loaded, missing: {missed}")
                # log_cuda_mem("after loading latent checkpoint", accelerator)
                torch.cuda.empty_cache()
                del missed
                gc.collect()
            else:
                logger.warning(f"Latent checkpoint {cfg.checkpoint.latent} passed, but not loaded, since soon it will be overwritten.")
                # log_cuda_mem("after loading latent empty cache", accelerator)
    # vae = load_LRM_vae()
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=cfg.num_workers)

    if not cfg.optimize:
        neural_brdf, latent_module, val_dataloader = accelerator.prepare(neural_brdf, latent_module, val_dataloader)
        load_state_if_any(output_path, accelerator, cfg.resume_path)

    else:

        sampler = SeedableRandomSampler(train_dataset, replacement=False, num_samples=cfg.n_iterations * train_dataset.n_mats)
        dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=8, sampler=sampler)

        latent_module.set_trainable(cfg.trainable.latent)
        neural_brdf.set_trainable(cfg.trainable.network)

        lr_scale = math.sqrt(cfg.batch_size * accelerator.num_processes) if cfg.scale_lr else 1.0
        trainable = []
        if cfg.trainable.network:
            trainable += [{"params": neural_brdf.parameters(), "lr": cfg.lr.network * lr_scale}]
            logger.info(f"Training network Learning rate: {cfg.lr.network * lr_scale} = {cfg.lr.network} * sqrt({cfg.batch_size} (batch_size) * {accelerator.num_processes} (GPU count))")

        if cfg.trainable.latent:
            trainable += [{
                    "params": latent_module.get_parameters(),
                    "lr": cfg.lr.latent * lr_scale,
                }
            ]
            logger.info(f"Training latent {latent_module.__class__.__name__} with learning rate: {cfg.lr.latent * lr_scale} = {cfg.lr.latent} * sqrt({cfg.batch_size} (batch_size) * {accelerator.num_processes} (GPU count))")

        optimizer_fn = getattr(torch.optim, cfg.optimizer)
        logger.info(f"Using optimizer {optimizer_fn}")

        optimizer = optimizer_fn(trainable)
        
        # Create scheduler
        scheduler = create_scheduler(cfg, optimizer, len(dataloader))
        logger.info(f"Using scheduler: {cfg.scheduler.name}")
        
        neural_brdf, latent_module, optimizer, dataloader, val_dataloader, scheduler = accelerator.prepare(
            neural_brdf, latent_module, optimizer, dataloader, val_dataloader, scheduler
        )

        # initialize metrics
        metrics = {
            "losses": [],
            "sigmas": [],
            "lrs": [],
            "lambdas": [],
            "step": -1,
        }
        metrics.update(load_state_if_any(output_path, accelerator, cfg.resume_path))

        n_iterations = len(dataloader)
        val_steps = max(n_iterations // 10, 10)
        half_life = n_iterations // 3

        step = metrics["step"] + 1
        dataloader = accelerator.skip_first_batches(dataloader, step)

        checkpoint_step = max(n_iterations // cfg.checkpoint_count, 100)
        n_iterations = len(dataloader) # after loading checkpoint, the total iteration count might change.
        logger.info(f"Number of iterations: {n_iterations}, initial step: {step}")

        # Initialize Falcor generator once per GPU (if using Falcor mode)
        # Use local_process_index for proper GPU selection (works with multi-node too)
        if cfg.use_falcor:
            from src.falcor_data import get_falcor_generator
            falcor_gen = get_falcor_generator(
                res=cfg.resolution,
                batch_shape=(64,64),
                spp=cfg.falcor.spp,
                log_level=cfg.falcor.log_level,
                process_index=accelerator.local_process_index,
            )
            falcor_gen.enable_cache = True # disable material caching
            logger.info(f"Falcor generator initialized for process {accelerator.local_process_index} (global: {accelerator.process_index})")
            
            # Timing accumulators for Falcor operations
            falcor_timing = {
                "load_material": 0.0,
                "generate_batch": 0.0,
                "parse_record": 0.0,
                "total": 0.0,
                "count": 0,
            }

        pbar = tqdm(total=n_iterations, disable=not accelerator.is_main_process)

        for _, data in enumerate(dataloader):
            if should_terminate():
                logger.info("Termination requested. Saving checkpoint...")
                save_and_requeue(accelerator, output_path, metrics, cfg.safe_tensors)
                logger.info("Checkpoint saved. Exiting.")
                return

            if step % 100 == 0:
                torch.cuda.empty_cache()

            sigma = int(math.exp(-step / half_life) * 4) * 2 + 1

            if (step) % val_steps == 0:
                validation(
                    cfg,
                    output_path=f"{output_path}/step_{step}",
                    metrics=metrics,
                    neural_brdf=neural_brdf,
                    latent_module=latent_module,
                    dl=val_dataloader,
                    val_dataset=val_dataset,
                    val_states=val_states,
                    accelerator=accelerator,
                    partial=True,
                    current_step=step,
                    lpips_fn=lpips_fn,
                )

            with accelerator.accumulate(neural_brdf, latent_module):
                if cfg.use_falcor:
                    # ========== FALCOR MODE: Direct BRDF supervision ==========
                    # Get batched material info from dataset
                    mat_id = data["mat_ids"]     # list of material indices
                    mat_names = data["names"]     # list of material names
                    rep_images = data.get("rep_images")
                    B = len(mat_names)
                    
                    # Generate Falcor samples for each material in batch
                    from src.falcor_data import parse_training_record
                    all_uv, all_wi, all_wo, all_target, all_valid, seeds = [], [], [], [], [], []
                    
                    falcor_step_start = time.perf_counter()
                    
                    for i in range(B):
                        # load_material now accepts torch tensors directly
                        # Pass material_name for caching - skips reload if same material
                        t0 = time.perf_counter()
                        falcor_gen.load_material(
                            basecolor=data['basecolor'][i],
                            roughness=data['roughness'][i],
                            metallic=data['metallic'][i],
                            height=data['height'][i],
                            normal=data['normal'][i] if 'normal' in data else None,
                            height_mean=float(data['height_mean'][i]),
                            height_factor=float(data['height_factor'][i]),
                            material_name=mat_names[i],  # Enable caching
                        )
                        t1 = time.perf_counter()
                        falcor_timing["load_material"] += t1 - t0
                        
                        seed = step * 7 + mat_id[i].item()*13 + 11*i + accelerator.process_index*17
                        seeds.append(seed)
                        
                        t2 = time.perf_counter()
                        raw_batch = falcor_gen.generate_batch(seed)
                        t3 = time.perf_counter()
                        falcor_timing["generate_batch"] += t3 - t2
                        
                        t4 = time.perf_counter()
                        batch = parse_training_record(raw_batch, mat_names[i], seed)
                        t5 = time.perf_counter()
                        falcor_timing["parse_record"] += t5 - t4
                        
                        # Don't unload - keep scene cached for next iteration
                        # falcor_gen.unload_material()
                        
                        all_uv.append(torch.from_numpy(batch.uv))
                        all_wi.append(torch.from_numpy(batch.wi))
                        all_wo.append(torch.from_numpy(batch.wo))
                        all_target.append(torch.from_numpy(batch.target))
                        all_valid.append(torch.from_numpy(batch.valid))
                    
                    falcor_step_end = time.perf_counter()
                    step_total = falcor_step_end - falcor_step_start
                    falcor_timing["total"] += step_total
                    falcor_timing["count"] += 1
                    
                    # Log timing periodically (every 100 steps)
                    if accelerator.is_main_process and step % 100 == 0:
                        n = falcor_timing["count"]
                        avg_total = 1000 * falcor_timing["total"] / n if n > 0 else 0
                        print(f"[Falcor] Step {step}: avg={avg_total:.1f}ms/step over {n} steps")
                    
                    # Stack and flatten with validity filter
                    uv = torch.stack(all_uv, dim=0).unsqueeze(1).to(device)
                    wi = torch.stack(all_wi, dim=0).unsqueeze(1).to(device)
                    wo = torch.stack(all_wo, dim=0).unsqueeze(1).to(device)
                    target = torch.stack(all_target, dim=0).unsqueeze(1).to(device)
                    valid = torch.stack(all_valid, dim=0).unsqueeze(1).to(device) > 0.5
                    
                    accelerator.unwrap_model(latent_module).blur_latents(mat_names, sigma)
                    
                    # Get latent texture for this material
                    checkpointing = (cfg.gradient_checkpointing, cfg.gradient_checkpointing_offload)
                    with accelerator.autocast():
                        latent_texture, offset_texture = latent_module(mat_names, rep_images, checkpointing=checkpointing)
                    
                    # Forward through neural BRDF
                    # fr, brdf_offset = neural_brdf(wi, wo, uv, latent_texture, offset_texture)
                    fr, brdf_offset = neural_brdf(wo, wi, uv, latent_texture, offset_texture)

                    # Loss: MSE between predicted and path-traced target
                    loss, _ = calc_loss(fr, target, cfg.loss_func, valid, None)
                    
                    if cfg.use_offset and cfg.offset_lambda is not None and brdf_offset is not None:
                        lambdaa = cfg.offset_lambda
                        metrics["lambdas"].append(lambdaa)
                        loss += lambdaa * brdf_offset.square().mean()
                    
                    if cfg.tv_loss is not None:
                        loss += (tv_loss(latent_texture) * cfg.tv_loss)
                        loss += (tv_loss(offset_texture) * cfg.tv_loss) if cfg.use_offset and offset_texture is not None else 0
                    
                    view_id = torch.tensor(seeds)
                    
                else:
                    # ========== IMAGE MODE: Original image-based training ==========
                    gt = data["images"]
                    mat_id = data["mat_ids"]
                    view_id = data["view_ids"]
                    mat_name = data["names"]
                    rep_images = data.get("rep_images")
                    states = [train_dataset.datasets[_mi].states for _mi in mat_id]

                    accelerator.unwrap_model(latent_module).blur_latents(mat_name, sigma)

                    checkpointing = (cfg.gradient_checkpointing, cfg.gradient_checkpointing_offload)
                    with accelerator.autocast():
                        latent_texture, offset_texture = latent_module(mat_name, rep_images, checkpointing=checkpointing)

                    outputs = render_torch(
                        states=states,
                        frame_ind=view_id,
                        neural_brdf=neural_brdf,
                        resolution=cfg.resolution,
                        device=device,
                        seed=step,
                        convert_to_point_light=True,
                        latent_texture=latent_texture,
                        offset_texture=offset_texture,
                        spp=cfg.spp,
                        jitter=cfg.jitter,
                    )

                    img_raw = outputs['image']
                    vis_mask = (outputs['visibility'] > 0.999999)[...,0].detach()

                    img = tonemapping(cfg.tonemapped, img_raw)

                    loss, _ = calc_loss(img, gt, cfg.loss_func, vis_mask, None)

                    if cfg.use_offset and cfg.offset_lambda is not None:
                        lambdaa = cfg.offset_lambda
                        metrics["lambdas"].append(lambdaa)
                        loss += lambdaa * (outputs['brdf_offset']).square().mean()

                    if cfg.tv_loss is not None:
                        loss += (tv_loss(latent_texture) * cfg.tv_loss)
                        loss += (tv_loss(offset_texture) * cfg.tv_loss) if cfg.use_offset else 0

                    if cfg.lpips.enabled:
                        lpips_loss_value = cfg.lpips.weight * lpips_loss(lpips_fn, img, gt, vis_mask)
                        if cfg.lpips.only_lpips:
                            loss = lpips_loss_value
                        else:
                            loss += lpips_loss_value

                accelerator.backward(loss)
                if cfg.max_grad_norm is not None and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(latent_module.parameters(), cfg.max_grad_norm)
                        accelerator.clip_grad_norm_(neural_brdf.parameters(), cfg.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                loss_values_this = accelerator.gather(loss).mean().detach().clone().item()
                current_lr = optimizer.param_groups[0]['lr']
                metrics["losses"].append(loss_values_this)
                metrics["sigmas"].append(sigma)
                metrics["lrs"].append(current_lr)
                metrics["step"] = step

                pbar.set_postfix(loss=f"{loss.item():.4f}", sigma=sigma, lr=f"{current_lr:.2e}", mat_id=mat_id.tolist(), view_id=view_id.tolist())
                pbar.update(1)
                save_state(checkpoint_step, accelerator, step, output_path, metrics, latent_module, cfg.safe_tensors)
                step += 1


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if cfg.trainable.latent:
            torch.save(accelerator.unwrap_model(latent_module).state_dict(), f"{output_path}/latent_module.pth")
        if cfg.trainable.network:
            torch.save(accelerator.unwrap_model(neural_brdf).state_dict(), f"{output_path}/mlp.pth")


    logger.info("Final validation")
    validation(
        cfg,
        output_path=f"{output_path}/final",
        metrics=metrics,
        neural_brdf=neural_brdf,
        latent_module=latent_module,
        dl=val_dataloader,
        val_dataset=val_dataset,
        val_states=val_states,
        accelerator=accelerator,
        current_step=-1,
        lpips_fn=lpips_fn,
    )


    mark_finished(accelerator, output_path)


if __name__ == "__main__":
    main()
