import sys
from typing import Any
import hydra
import gc

from utils import remove_module_prefix, save_state, load_state_if_any, log_cuda_mem, is_resumption
from signal_handling import should_terminate, save_and_requeue, mark_finished

sys.path.append("./")
import numpy as np
import torch

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
    init_lpips,
    lpips_loss,
)
import math
from src.data import load_datasets
from torch.utils.data import RandomSampler, DataLoader
from accelerate import Accelerator, DistributedType
from accelerate.utils import DeepSpeedPlugin
from accelerate.data_loader import SeedableRandomSampler
from accelerate.logging import get_logger
from src.models.wan_lrm import load_LRM_vae
from pathlib import Path

logger = get_logger(__name__)

# Enable verbose DeepSpeed logging to see sharding info
os.environ["DEEPSPEED_LOG_LEVEL"] = "INFO"

torch.cuda.set_per_process_memory_fraction(0.98, device=0)


class UnifiedModel(torch.nn.Module):
    """
    Wrapper that combines latent_module and neural_brdf into a single model.
    Required for DeepSpeed which only supports a single model in accelerator.prepare().
    """
    def __init__(self, latent_module, neural_brdf):
        super().__init__()
        self.latent_module = latent_module
        self.neural_brdf = neural_brdf
    
    def set_trainable(self, train_latent: bool, train_network: bool):
        self.latent_module.set_trainable(train_latent)
        self.neural_brdf.set_trainable(train_network)
    
    def get_latent_parameters(self):
        return self.latent_module.get_parameters()
    
    def get_network_parameters(self):
        return list(self.neural_brdf.parameters())
    
    def blur_latents(self, mat_name, sigma):
        self.latent_module.blur_latents(mat_name, sigma)
    
    def forward_latent(self, mat_name, rep_images, checkpointing):
        return self.latent_module(mat_name, rep_images, checkpointing=checkpointing)
    
    def forward(self, *args, **kwargs):
        # Not used directly - we call forward_latent and neural_brdf separately
        raise NotImplementedError("Use forward_latent and neural_brdf directly")


@hydra.main(version_base=None, config_path="config", config_name="inverse_render")
def main(cfg: DictConfig) -> None:
    output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    # Configure DeepSpeed Zero3 with CPU offloading - shards model params across GPUs
    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=3,
        offload_optimizer_device="cpu",
        offload_param_device="cpu",  # Also offload params to CPU when not in use
        zero3_init_flag=True,  # Required for Zero3 to properly initialize sharded params
        zero3_save_16bit_model=True,  # Save consolidated fp16 model for easier loading
        gradient_accumulation_steps=cfg.grad_acc_steps,
    )
    
    # Log DeepSpeed config for visibility
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print("\n" + "="*60)
        print("DeepSpeed Zero3 Configuration:")
        print("  - Model parameters: SHARDED across GPUs")
        print("  - Optimizer states: SHARDED + CPU offload")
        print("  - Gradients: SHARDED")
        print("="*60 + "\n")
    
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        gradient_accumulation_steps=cfg.grad_acc_steps,
        deepspeed_plugin=deepspeed_plugin,
    )
    
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
            "network_width": cfg.network.width,
            "network_depth": cfg.network.depth,
        }
    )
    logger.info(f"Rank {accelerator.process_index}, {accelerator.local_process_index}: MLP loaded", main_process_only=False)

    # Create unified model for DeepSpeed compatibility
    unified_model = UnifiedModel(latent_module, neural_brdf)
    logger.info(f"Rank {accelerator.process_index}: Created unified model for DeepSpeed", main_process_only=False)

    # Initialize LPIPS if enabled
    if cfg.lpips.enabled:
        lpips_fn = init_lpips(cfg.lpips.net, device)
        logger.info(f"LPIPS initialized with {cfg.lpips.net} backbone, weight={cfg.lpips.weight}")
    else:
        lpips_fn = None

    with accelerator.local_main_process_first():
        if cfg.checkpoint.network is not None:
            missed = unified_model.neural_brdf.load_state_dict(remove_module_prefix(torch.load(cfg.checkpoint.network, map_location=device)), strict=False)
            logger.info(f"Network checkpoint loaded, missing: {missed}")
            del missed
            gc.collect()
        if cfg.checkpoint.latent is not None:
            logger.info(f"Loading a checkpoint from {cfg.checkpoint.latent}")
            if is_resumption(output_path, cfg.resume_path)[0] == 0:
                missed = unified_model.latent_module.load_state_dict(remove_module_prefix(torch.load(cfg.checkpoint.latent, map_location=device)), strict=False)
                logger.info(f"Latent checkpoint loaded, missing: {missed}")
                # log_cuda_mem("after loading latent checkpoint", accelerator)
                torch.cuda.empty_cache()
                del missed
                gc.collect()
            else:
                logger.warning(f"Latent checkpoint {cfg.checkpoint.latent} passed, but not loaded, since soon it will be overwritten.")
    # vae = load_LRM_vae()
    val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=cfg.num_workers)

    if not cfg.optimize:
        unified_model, val_dataloader = accelerator.prepare(unified_model, val_dataloader)
        load_state_if_any(output_path, accelerator, cfg.resume_path)

    else:

        sampler = SeedableRandomSampler(train_dataset, replacement=False, num_samples=cfg.n_iterations * train_dataset.n_mats)
        dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=8, sampler=sampler)

        unified_model.set_trainable(cfg.trainable.latent, cfg.trainable.network)

        lr_scale = math.sqrt(cfg.batch_size * accelerator.num_processes) if cfg.scale_lr else 1.0
        trainable = []
        if cfg.trainable.network:
            trainable += [{"params": unified_model.get_network_parameters(), "lr": cfg.lr.network * lr_scale}]
            logger.info(f"Training network Learning rate: {cfg.lr.network * lr_scale} = {cfg.lr.network} * sqrt({cfg.batch_size} (batch_size) * {accelerator.num_processes} (GPU count))")

        if cfg.trainable.latent:
            trainable += [{
                    "params": unified_model.get_latent_parameters(),
                    "lr": cfg.lr.latent * lr_scale,
                }
            ]
            logger.info(f"Training latent with learning rate: {cfg.lr.latent * lr_scale} = {cfg.lr.latent} * sqrt({cfg.batch_size} (batch_size) * {accelerator.num_processes} (GPU count))")

        optimizer_fn = getattr(torch.optim, cfg.optimizer)
        logger.info(f"Using optimizer {optimizer_fn}")

        optimizer = optimizer_fn(trainable)
        
        # Create scheduler
        scheduler = create_scheduler(cfg, optimizer, len(dataloader))
        logger.info(f"Using scheduler: {cfg.scheduler.name}")
        
        # DeepSpeed: only pass one model to prepare()
        unified_model, optimizer, dataloader, val_dataloader, scheduler = accelerator.prepare(
            unified_model, optimizer, dataloader, val_dataloader, scheduler
        )
        
        # Log Zero3 sharding info
        if accelerator.is_main_process:
            print("\n" + "="*60)
            print("DeepSpeed Zero3 Model Sharding Info:")
            print("="*60)
            
            # Get the DeepSpeed engine
            ds_engine = unified_model
            if hasattr(ds_engine, 'module'):
                total_params = sum(p.ds_numel if hasattr(p, 'ds_numel') else p.numel() 
                                   for p in ds_engine.module.parameters())
                local_params = sum(p.ds_tensor.numel() if hasattr(p, 'ds_tensor') else p.numel() 
                                   for p in ds_engine.module.parameters())
                print(f"  Total model parameters: {total_params:,}")
                print(f"  Parameters per GPU (sharded): {local_params:,}")
                print(f"  Sharding factor: {total_params / max(local_params, 1):.1f}x")
                print(f"  Number of GPUs: {accelerator.num_processes}")
                print(f"  Memory per GPU (params only): ~{local_params * 4 / 1e9:.2f} GB (fp32)")
            print("="*60 + "\n")

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

        pbar = tqdm(total=n_iterations, disable=not accelerator.is_main_process)

        for _, data in enumerate(dataloader):
            if should_terminate():
                logger.info("Termination requested. Saving checkpoint...")
                save_and_requeue(accelerator, output_path, metrics, cfg.safe_tensors)
                # if accelerator.is_main_process:
                #     torch.save(accelerator.unwrap_model(latent_module).state_dict(), f"{output_path}/states/latent_module.pth")
                logger.info("Checkpoint saved. Exiting.")
                return
            gt = data["images"]
            mat_id = data["mat_ids"]
            view_id = data["view_ids"]
            mat_name = data["names"]
            rep_images = data.get("rep_images")
            states = [train_dataset.datasets[_mi].states for _mi in mat_id]

            if step % 100 == 0:
                torch.cuda.empty_cache()

            sigma = int(math.exp(-step / half_life) * 4) * 2 + 1

            # if (step) % val_steps == 0:
            #     validation(
            #         cfg,
            #         output_path=f"{output_path}/step_{step}",
            #         metrics=metrics,
            #         neural_brdf=neural_brdf,
            #         latent_module=latent_module,
            #         dl=val_dataloader,
            #         val_dataset=val_dataset,
            #         val_states=val_states,
            #         accelerator=accelerator,
            #         partial=True,
            #     )

            accelerator.unwrap_model(unified_model).blur_latents(mat_name, sigma)

            # log_cuda_mem(f"before latent forward (step={step}, rank={accelerator.process_index})", accelerator)

            with accelerator.accumulate(unified_model):
                checkpointing = (cfg.gradient_checkpointing, cfg.gradient_checkpointing_offload)
                with accelerator.autocast():
                    latent_texture, offset_texture = accelerator.unwrap_model(unified_model).forward_latent(mat_name, rep_images, checkpointing)
                if step % 2 == 0:
                    torch.cuda.empty_cache()


                outputs = render_torch(
                    states=states,
                    frame_ind=view_id,
                    neural_brdf=accelerator.unwrap_model(unified_model).neural_brdf,
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
                vis_mask = (outputs['visibility'] > 0.999999)[...,0].detach() # shape = H,w


                img = tonemapping(cfg.tonemapped, img_raw)

                loss, _ = calc_loss(img, gt, cfg.loss_func, vis_mask, None)

                if cfg.use_offset and cfg.offset_lambda is not None:
                    lambdaa = cfg.offset_lambda
                    metrics["lambdas"].append(lambdaa)
                    loss+=lambdaa*(outputs['brdf_offset']).square().mean()

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
                        accelerator.clip_grad_norm_(unified_model.parameters(), cfg.max_grad_norm)

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
                save_state(checkpoint_step, accelerator, step, output_path, metrics, unified_model, cfg.safe_tensors)
                step += 1


    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(unified_model)
    if accelerator.distributed_type in (DistributedType.FSDP, DistributedType.DEEPSPEED):
        accelerator.save_state(f"{output_path}/accelerator_state")
        logger.info(f"Saved accelerator state to {output_path}/accelerator_state")
    
    # Also save individual model checkpoints for inference
    if accelerator.is_main_process:
        if cfg.trainable.latent:
            torch.save(unwrapped.latent_module.state_dict(), f"{output_path}/latent_module.pth")
        if cfg.trainable.network:
            torch.save(unwrapped.neural_brdf.state_dict(), f"{output_path}/mlp.pth")
        logger.info(f"Saved model checkpoints to {output_path}")


    logger.info("Final validation")
    # validation(
    #     cfg,
    #     output_path=f"{output_path}/final",
    #     metrics=metrics,
    #     neural_brdf=unwrapped.neural_brdf,
    #     latent_module=unwrapped.latent_module,
    #     dl=val_dataloader,
    #     val_dataset=val_dataset,
    #     val_states=val_states,
    #     accelerator=accelerator,
    # )


    mark_finished(accelerator, output_path)


if __name__ == "__main__":
    main()
