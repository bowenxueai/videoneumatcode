import sys
from typing import Any
import hydra
import gc

from utils import remove_module_prefix, save_state, load_state_if_any, log_cuda_mem, is_resumption
from signal_handling import should_terminate, save_and_requeue, mark_finished

sys.path.append("./")
import numpy as np
import torch
from PIL import Image
from src.neumip_nets import initalize_network, initalize_latents
import os
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from common import (
    seed_everything,
    tonemapping,
    calc_loss,
    validation,
    render_torch,
    load_gts,
    load_scene_states,
    tv_loss,
    create_scheduler,
)
import math
from src.data import load_datasets
from torch.utils.data import RandomSampler, DataLoader
from accelerate import Accelerator
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
        )

        logger.info(f"Train dataset: {train_dataset.summary()}")

        val_dataset = load_datasets(
            tonemapping_type=cfg.tonemapped,
            unrolled=False,
            resolution=cfg.resolution,
            disable_rep=cfg.disable_rep_images_loading,
            rep_use_latent=cfg.latent.receives_vae_latent,
            cfgs=cfg.val_dataset,
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
                "num_materials": train_dataset.n_mats,
                "device": device,
                "input_resolution": cfg.resolution,
                "n_input_frames": len(train_dataset.datasets[0].representative_frame_list),
            },
        )
    logger.info(f"Rank {accelerator.process_index}, {accelerator.local_process_index}: Latent module loaded", main_process_only=False)
    neural_brdf = initalize_network(
        {
            "use_offset": cfg.use_offset,
            "latent_channels": cfg.latent.channels,
            "min_z": cfg.network.min_z,
            "network_width": cfg.network.width,
            "network_depth": cfg.network.depth,
        }
    )
    logger.info(f"Rank {accelerator.process_index}, {accelerator.local_process_index}: MLP loaded", main_process_only=False)

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
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=8)

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


        n_iterations = len(dataloader) # after loading checkpoint, the total iteration count might change.
        checkpoint_step = max(n_iterations // 500, 10)
        logger.info(f"Number of iterations: {n_iterations}, initial step: {step}")

        pbar = tqdm(total=n_iterations, disable=not accelerator.is_main_process)

        for _, data in enumerate(dataloader):
            if should_terminate():
                logger.info("Termination requested. Saving checkpoint...")
                save_and_requeue(accelerator, output_path, metrics)
                if accelerator.is_main_process:
                    torch.save(accelerator.unwrap_model(latent_module).state_dict(), f"{output_path}/latent_module.pth")
                logger.info("Checkpoint saved. Exiting.")
                return
            gt = data["images"]
            mat_id = data["mat_ids"]
            view_id = data["view_ids"]
            rep_images = data.get("rep_images")
            states = [train_dataset.datasets[_mi].states for _mi in mat_id]

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
                )

            accelerator.unwrap_model(latent_module).blur_latents(mat_id, sigma)

            # log_cuda_mem(f"before latent forward (step={step}, rank={accelerator.process_index})", accelerator)

            with accelerator.accumulate(neural_brdf, latent_module):
                checkpointing = (cfg.gradient_checkpointing, cfg.gradient_checkpointing_offload)
                with accelerator.autocast():
                    latent_texture, offset_texture = latent_module(mat_id, rep_images, checkpointing=checkpointing)

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
                )



                img_raw = outputs['image']
                vis_mask = (outputs['visibility'] > 0.999999)[...,0].detach() # shape = H,w


                img = tonemapping(cfg.tonemapped, img_raw)

                loss, loss_img = calc_loss(img, gt, cfg.loss_func, vis_mask, None)



                if accelerator.is_main_process and step % 100 == 0:
                    os.makedirs(f"{output_path}/{step}", exist_ok=True)

                    for i in range(len(img)):
                        os.makedirs(f"{output_path}/{step}/{i}", exist_ok=True)

                        Image.fromarray((img[i].detach().cpu().numpy()*255).astype(np.uint8)).save(f"{output_path}/{step}/{i}/img_{step}_{i}.png")
                        Image.fromarray((vis_mask[i].detach().cpu().numpy()*255).astype(np.uint8)).save(f"{output_path}/{step}/{i}/vmask_{step}_{i}.png")
                        Image.fromarray(((loss_img[i].detach().cpu().numpy()*10)*255).astype(np.uint8)).save(f"{output_path}/{step}/{i}/loss_img_{step}_{i}.png")
                        gt_masked = gt[i].clone()
                        gt_masked[~vis_mask[i]] = 0
                        print("gt", gt[i].shape, "gt_masked", gt_masked.shape, vis_mask[i].shape)
                        Image.fromarray(((gt[i].detach().cpu().numpy()*255).astype(np.uint8))).save(f"{output_path}/{step}/{i}/gt_{step}_{i}.png")
                        Image.fromarray(((gt_masked.detach().cpu().numpy()*255).astype(np.uint8))).save(f"{output_path}/{step}/{i}/gt_masked_{step}_{i}.png")
                        
                        




                if cfg.use_offset and cfg.offset_lambda is not None:
                    lambdaa = cfg.offset_lambda
                    metrics["lambdas"].append(lambdaa)
                    loss+=lambdaa*(outputs['brdf_offset']).square().mean()

                if cfg.tv_loss is not None:
                    loss += (tv_loss(latent_texture) * cfg.tv_loss)
                    loss += (tv_loss(offset_texture) * cfg.tv_loss) if cfg.use_offset else 0

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
                save_state(checkpoint_step, accelerator, step, output_path, metrics, latent_module)
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
    )


    mark_finished(accelerator, output_path)


if __name__ == "__main__":
    main()
