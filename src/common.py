import os
import json
import glob
from collections import defaultdict
import torch
from transformers import get_scheduler
import numpy as np
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import drjit as dr
from PIL import Image
from src.utils import colormap_diverging, unpack_video_to_numpy, save_video, plot_loss
from src.neumip_nets import initalize_network, initalize_latents
from omegaconf import OmegaConf
from pathlib import Path
import pyexr
import math
from src.torch_renderer import (
    Plane,
    Texture,
    create_camera_from_mitsuba,
    DiffuseBRDF,
    render,
)
from accelerate.logging import get_logger
logger = get_logger(__name__)


def create_scheduler(cfg, optimizer, num_training_steps: int):
    """Create learning rate scheduler based on config.
    
    Uses transformers.get_scheduler. Supported scheduler names:
    - constant: No warmup, constant LR
    - cosine: Cosine decay with linear warmup (recommended for transformers)
    - linear: Linear decay with linear warmup
    - constant_with_warmup: Constant LR after linear warmup
    - polynomial: Polynomial decay with linear warmup
    - cosine_with_restarts: Cosine with hard restarts
    - inverse_sqrt: Inverse square root decay
    """
    scheduler_name = cfg.scheduler.name
    warmup_ratio = cfg.scheduler.get("warmup_ratio", 0.0)
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    
    return get_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)



def validation_cameras(total_n_frames= 320, area_light_radius = 0.25, lightgeight = 2, camera_height = 4 , base_light_intensity = [204.5762567336908, 204.5762567336908, 204.5762567336908], camera_radius = 4.0, use_area_light = True):
    params = []
    n_frames = total_n_frames//2

    t0 = np.linspace(-20, 20, n_frames//2) # hardcoded 20
    t1 = np.linspace(20, -3, n_frames - n_frames//2)
    ty = np.concatenate([t0, 0*t1])
    tx = np.concatenate([0*t0, t1])

    lightpos = np.stack([
        tx,
        ty,
        tx*0+(lightgeight * 2 + 3)
    ], axis = -1)



    actual_distance = np.sqrt(np.sum(lightpos**2, axis = -1))
    base_distance = lightgeight * 2+2


    light_energy_pname = 'area_light_sphere.emitter.radiance.value' if use_area_light else 'light.intensity.value'
    light_loc_pname = 'area_light_sphere.to_world' if use_area_light else 'light.position'

    intensity_correction = (actual_distance / base_distance) ** 2
    corrected_intensity = np.array(base_light_intensity)[None] * intensity_correction[:,None]
    last_light_energy = None
    last_light_loc = None

    for i in range(n_frames):
        params.append({})
        last_light_energy = corrected_intensity[i]
        last_light_loc = (mi.Transform4f().translate(lightpos[i]).scale(area_light_radius)).matrix.numpy().tolist() if use_area_light else lightpos[i].tolist()
        params[-1][light_energy_pname] = last_light_energy
        params[-1][light_loc_pname] = last_light_loc
        params[-1]['sensor.to_world'] = mi.Transform4f().look_at(
                                        origin=[0,0,camera_height+2],
                                        target=[0, 0, 0],
                                        up=[0, 1, 0],
                                    ).matrix.numpy().tolist()
        params[-1]['sensor.x_fov'] = 30
                                    



    n_frames = total_n_frames - total_n_frames//2
    # n_frames = total_n_frames

    t = np.linspace(0.5, 1.5, n_frames)
    camera_pos = np.stack([
        camera_radius*0.4 * np.cos(t * 2*np.pi),
        camera_radius*0.4 * np.sin(t * 2*np.pi),
        (np.cos((t) *np.pi)*camera_height+camera_height+2)
    ], axis = -1)

    camera_pos[:,0] += camera_radius*0.4   # to bring the first location of the camera to the location in the first phase: [0,0,camera_height + 2]


    for i in range(n_frames):
        params.append({})
        params[-1][light_energy_pname] = last_light_energy
        params[-1][light_loc_pname] = last_light_loc
        params[-1]['sensor.to_world'] = mi.Transform4f().look_at(
                                        origin=camera_pos[i].tolist(),
                                        target=[0, 0, 0],
                                        up=[0, 1- t[i].item(), t[i].item()],
                                    ).matrix.numpy().tolist()
        params[-1]['sensor.x_fov'] = 30

    return params



def tonemapping(tonemapping_type, img):
    np_in = isinstance(img, np.ndarray)
    if np_in:
        img = torch.from_numpy(img)
    def _linear_to_srgb_gpu(x, clamp = True):

        a = 0.055
        if clamp:
            x_pos = x.clamp(min = 0.0)
        else:
            x_pos = x
        y = 12.92 * x_pos
        y[x_pos > 0.0031308] = (1 + a) * (x_pos[x_pos > 0.0031308].pow(1.0 / 2.4)) - a

        if clamp:
            y = y.clamp(min = 0.0, max = 1.0)            
        return y

    out = None

    if tonemapping_type == 'srgb':
        out = _linear_to_srgb_gpu(img)
    elif tonemapping_type == 'srgb_no_clamp':
        out = _linear_to_srgb_gpu(img, clamp = False)
    elif tonemapping_type == 'reinhard':
        coeff = 7.0
        out = coeff * img / (coeff * img + 1)
    elif tonemapping_type == 'none':
        out=img
    else:
        raise ValueError(f"Invalid tonemapping: {tonemapping_type}")

    if np_in:
        out = out.numpy()
    return out



def calc_loss(image, image_ref, loss_func, masks, weights):
    if loss_func.lower() == 'l2':
        loss_img =  (image - image_ref)**2
    elif loss_func.lower() == 'l2rel':
        loss_img = (image - image_ref)**2 / ((image_ref**2).detach() + 0.01)

    elif loss_func.lower() == 'l2log':
        loss_img = (torch.log(1+image) - torch.log(1+torch.detach(image_ref)))**2
    else:
        raise Exception('loss not recognized')

    if weights is not None:
        loss_img = torch.detach(weights) * loss_img

    if masks is None:
        return torch.mean(loss_img), loss_img
    else:

        # assert masks.ndim == 2

        loss = (loss_img[masks]).mean()

        

    return loss, loss_img


def init_lpips(net='vgg', device='cuda'):
    """Initialize LPIPS perceptual loss model.
    
    Args:
        net: Network backbone - 'vgg', 'alex', or 'squeeze'
        device: Device to place the model on
    
    Returns:
        LPIPS model in eval mode
    """
    import lpips
    lpips_fn = lpips.LPIPS(net=net).to(device)
    lpips_fn.eval()
    for param in lpips_fn.parameters():
        param.requires_grad = False
    return lpips_fn


def lpips_loss(lpips_fn, image, image_ref, mask=None):
    """Compute LPIPS perceptual loss between two images.
    
    Args:
        lpips_fn: Initialized LPIPS model from init_lpips()
        image: Predicted image, shape [B, H, W, 3] in [0, 1]
        image_ref: Reference image, shape [B, H, W, 3] in [0, 1]
        mask: Optional boolean mask, shape [B, H, W] or [H, W]. 
              If provided, masks both images before computing LPIPS.
    
    Returns:
        Scalar LPIPS loss
    """
    # Apply mask if provided (zero out background in both images)
    if mask is not None:
        mask_expanded = mask.unsqueeze(-1) if mask.ndim == image.ndim - 1 else mask
        image = image * mask_expanded
        image_ref = image_ref * mask_expanded
    
    # Convert from [0, 1] to [-1, 1]
    image = image * 2 - 1
    image_ref = image_ref * 2 - 1
    
    # Permute from [B, H, W, C] to [B, C, H, W]
    image = image.permute(0, 3, 1, 2)
    image_ref = image_ref.permute(0, 3, 1, 2)
    
    # Downsample large images to avoid OOM in LPIPS VGG network
    max_lpips_size = 1024
    _, _, H, W = image.shape
    if H > max_lpips_size or W > max_lpips_size:
        scale = max_lpips_size / max(H, W)
        new_H, new_W = int(H * scale), int(W * scale)
        image = torch.nn.functional.interpolate(image, size=(new_H, new_W), mode='bilinear', align_corners=False)
        image_ref = torch.nn.functional.interpolate(image_ref, size=(new_H, new_W), mode='bilinear', align_corners=False)
    
    return lpips_fn(image, image_ref).mean()


def tv_loss(x):
    return (x[..., :-1, :] - x[..., 1:, :]).abs().mean() + (x[..., :, :-1] - x[..., :, 1:]).abs().mean()


def compute_image_metrics(rendered, gt, lpips_fn=None):
    """Compute multiple error metrics between rendered and ground truth images.
    
    Args:
        rendered: Rendered image as numpy array, shape [H, W, 3] in [0, 1] range
        gt: Ground truth image as numpy array or torch tensor, shape [H, W, 3] in [0, 1] range
        lpips_fn: Optional initialized LPIPS model for perceptual loss
    
    Returns:
        dict with keys: 'mse', 'psnr', 'mape', and optionally 'lpips'
    """
    # Convert gt to numpy if it's a tensor
    if hasattr(gt, 'cpu'):
        gt = gt.cpu().detach().numpy()
    
    # Ensure both are float arrays
    rendered = np.asarray(rendered, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    
    # MSE
    mse = ((rendered - gt) ** 2).mean()
    
    # PSNR (assuming images are in [0, 1] range, so max value is 1.0)
    psnr = -10 * np.log10(mse + 1e-10)
    
    # MAPE (Mean Absolute Percentage Error with epsilon for stability)
    mape = (((gt - rendered) ** 2) / (gt ** 2 + 0.01)).mean()
    
    metrics = {
        'mse': float(mse),
        'psnr': float(psnr),
        'mape': float(mape),
    }
    
    # LPIPS (requires torch tensors)
    if lpips_fn is not None:
        # Convert to torch tensors and add batch dimension
        rendered_t = torch.from_numpy(rendered).unsqueeze(0).float()
        gt_t = torch.from_numpy(gt).unsqueeze(0).float()
        
        # Move to same device as lpips_fn
        device = next(lpips_fn.parameters()).device
        rendered_t = rendered_t.to(device)
        gt_t = gt_t.to(device)
        
        with torch.no_grad():
            lpips_val = lpips_loss(lpips_fn, rendered_t, gt_t)
        metrics['lpips'] = float(lpips_val.cpu().numpy())
    
    return metrics


def compute_batch_metrics(renderings, gts, lpips_fn=None):
    """Compute metrics for a batch of images.
    
    Args:
        renderings: List of rendered images as numpy arrays, each [H, W, 3] in [0, 1]
        gts: List of ground truth images (numpy or torch tensors), each [H, W, 3] in [0, 1]
        lpips_fn: Optional initialized LPIPS model
    
    Returns:
        dict with per-view metrics: {view_idx: {'mse': ..., 'psnr': ..., 'mape': ..., 'lpips': ...}}
    """
    all_metrics = {}
    for i in range(len(renderings)):
        gt_i = gts[i].cpu().detach().numpy() if hasattr(gts[i], 'cpu') else gts[i]
        all_metrics[i] = compute_image_metrics(renderings[i], gt_i, lpips_fn)
    return all_metrics


def collect_errors_from_jsons(error_dir):
    """Collect and aggregate error metrics from JSON files in error_dir.
    
    Args:
        error_dir: Path to directory containing error JSON files.
                   Files should be named: {process_index}_{mat_name}_{split}_{traj}.json
                   Each file contains: {view_idx: {'mse': ..., 'psnr': ..., 'mape': ..., 'lpips': ...}}
    
    Returns:
        dict: Summary with 'per_split_traj' and 'overall' error statistics for each metric type.
    """
    # Collect unique {mat_name}_{split}_{traj} combinations
    seen_combinations = {}
    # Use glob.escape to handle special characters (like []) in directory path
    for json_file in glob.glob(f'{glob.escape(error_dir)}/*.json'):
        if json_file.endswith('summary.json'):
            continue
        filename = os.path.basename(json_file)
        # Filename format: {process_index}_{mat_name}_{split}_{traj}.json
        parts = filename.replace('.json', '').split('_', 1)  # Split off process_index
        assert len(parts) >= 2, f"Malformed error filename: {filename}, expected format: {{process_index}}_{{mat_name}}_{{split}}_{{traj}}.json"
        combo_key = parts[1]  # {mat_name}_{split}_{traj}
        if combo_key not in seen_combinations:
            seen_combinations[combo_key] = json_file
    
    # Metric types to aggregate
    metric_types = ['mse', 'psnr', 'mape', 'lpips']
    
    # Group by split_traj and compute averaged errors for each metric type
    split_errors = {m: defaultdict(list) for m in metric_types}
    all_errors = {m: [] for m in metric_types}
    
    for combo_key, json_file in seen_combinations.items():
        with open(json_file, 'r') as f:
            error_data = json.load(f)
        
        # Compute mean for each metric type for this material
        mat_metrics = {m: [] for m in metric_types}
        for view_key, view_metrics in error_data.items():
            if isinstance(view_metrics, dict):
                # New format: {view_idx: {'mse': ..., 'psnr': ..., ...}}
                for m in metric_types:
                    if m in view_metrics:
                        mat_metrics[m].append(float(view_metrics[m]))
            else:
                # Legacy format: {view_idx: mse_value}
                mat_metrics['mse'].append(float(view_metrics))
        
        # Extract split_traj from combo_key (format: {mat_name}_{split}_{traj})
        parts = combo_key.rsplit('_', 2)
        assert len(parts) >= 3, f"Malformed combo_key: {combo_key}, expected format: {{mat_name}}_{{split}}_{{traj}}"
        split_traj = f"{parts[-2]}_{parts[-1]}"  # e.g., "test_sparse"
        
        for m in metric_types:
            if mat_metrics[m]:
                avg_val = np.mean(mat_metrics[m])
                split_errors[m][split_traj].append(avg_val)
                all_errors[m].append(avg_val)
    
    # Build summary
    summary = {'per_split_traj': {}, 'overall': {}}
    
    # Get all unique split_trajs
    all_split_trajs = set()
    for m in metric_types:
        all_split_trajs.update(split_errors[m].keys())
    
    for split_traj in all_split_trajs:
        summary['per_split_traj'][split_traj] = {}
        for m in metric_types:
            if split_errors[m][split_traj]:
                errors = split_errors[m][split_traj]
                summary['per_split_traj'][split_traj][m] = {
                    'mean': float(np.mean(errors)),
                    'std': float(np.std(errors)),
                    'count': len(errors)
                }
    
    # Overall averaged errors for each metric
    assert len(all_errors['mse']) > 0, f"No error data found in {error_dir}"
    for m in metric_types:
        if all_errors[m]:
            summary['overall'][m] = {
                'mean': float(np.mean(all_errors[m])),
                'std': float(np.std(all_errors[m])),
                'count': len(all_errors[m])
            }
    
    return summary


def collect_and_update_error_history(error_dir, run_root, step_folder, current_step=None):
    """Collect errors from JSONs, update error history, save summary and plot.
    
    Args:
        error_dir: Path to directory containing error JSON files for this step.
        run_root: Path to run's root directory (for error_history.json).
        step_folder: Path to current step folder (for validation_error.png).
        current_step: Current training step (for x-axis of plot).
    """
    error_history_path = f'{run_root}/error_history.json'
    metric_types = ['mse', 'psnr', 'mape', 'lpips']
    
    # Load or initialize error history
    if os.path.exists(error_history_path):
        with open(error_history_path, 'r') as f:
            error_history = json.load(f)
    else:
        error_history = {
            'iterations': [],
            'overall': {m: [] for m in metric_types},
            'per_split_traj': {}
        }
    
    if os.path.exists(error_dir):
        summary = collect_errors_from_jsons(error_dir)
        
        with open(f'{error_dir}/summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Append to error history and save
        iter_val = current_step if current_step is not None else len(error_history['iterations'])
        error_history['iterations'].append(iter_val)
        
        # Append overall metrics
        for m in metric_types:
            if m in summary['overall']:
                if m not in error_history['overall']:
                    error_history['overall'][m] = []
                error_history['overall'][m].append(summary['overall'][m]['mean'])
        
        # Append per split_traj metrics
        for split_traj, split_data in summary['per_split_traj'].items():
            if split_traj not in error_history['per_split_traj']:
                error_history['per_split_traj'][split_traj] = {m: [] for m in metric_types}
            for m in metric_types:
                if m in split_data:
                    if m not in error_history['per_split_traj'][split_traj]:
                        error_history['per_split_traj'][split_traj][m] = []
                    error_history['per_split_traj'][split_traj][m].append(split_data[m]['mean'])
        
        with open(error_history_path, 'w') as f:
            json.dump(error_history, f, indent=2)
        
        # Plot aggregated MSE error over iterations
        if error_history['overall'].get('mse'):
            plot_fig = plot_loss(
                np.asarray(error_history['overall']['mse']),
                error_history['iterations'],
                resolution=(800, 800),
                ylabel='Val MSE',
                xlabel='Iteration',
                legend=' ',
                color='blue',
                font_size=None,
                linewidth=2,
                y_log_scale=True
            )
            plot_fig.save(f'{step_folder}/validation_error.png')


def validation(cfg, dl, metrics, neural_brdf, latent_module, output_path, val_dataset, val_states, accelerator, partial = False, current_step = None, lpips_fn = None):
    accelerator.wait_for_everyone()
    neural_brdf.eval()
    latent_module.eval()

    out_folder = output_path

    
    device = accelerator.device
    for b_id, batch in enumerate(dl):
        do_val_views = True
        # partial = False
        logger.info(f"Validation: device {accelerator.process_index}, batches: {b_id}, mat_names: {batch['names']}", main_process_only=False)
        mat_name = batch['names'][0]
        split = batch['split'][0]
        traj = batch['traj'][0]
        output_path = f'{out_folder}/{split}_split/{traj}_traj/{mat_name}'
        if Path(output_path).exists():
            logger.info(f"validation already done for {output_path}")
            continue
        os.makedirs(output_path, exist_ok=True)


        mat_id = batch['mat_ids'][0]
        rep_images = batch.get('rep_images')
        train_states = val_dataset.datasets[mat_id].states
        train_view_ids = batch['view_ids'].tolist()[0]
        train_gts = batch['images'][0]


        val_gts = None
        val_view_ids = list(range(len(val_states)))

        render_bs=10
        frames_to_save = {"train": [0], "val": [0,196,239]}

        if partial:
            val_view_ids = frames_to_save["val"]
            train_view_ids = frames_to_save["train"]
            frames_to_save = {"train": list(range(len(frames_to_save['train']))), "val": list(range(len(frames_to_save['val'])))}
            render_bs = 1

        


        with torch.no_grad():
            latents, offset_latents = latent_module([mat_name], rep_images, [False, False])

        for state_list, name_prefix, gts, view_ids in zip([train_states, val_states], ['train', 'val'], [train_gts, val_gts], [train_view_ids, val_view_ids]):
            if 'val' in name_prefix and not do_val_views:
                continue
            torch.cuda.empty_cache()
            renderings, offsets = [], []
            n_frames = len(view_ids)
            for i in range(0, n_frames, render_bs):
                with torch.no_grad():
                    this_bs = len(view_ids[i:i+render_bs])
                    
                    output = render_torch(states=[state_list]*this_bs, frame_ind=view_ids[i:i+this_bs], neural_brdf=neural_brdf, resolution=cfg.resolution, device=device, seed=0, spp=cfg.spp, convert_to_point_light=True, latent_texture=latents.expand(this_bs, -1, -1, -1), offset_texture=offset_latents.expand(this_bs, -1, -1, -1), jitter=cfg.jitter)

                for j in range(this_bs):

                    img = output['image'][j]
                    brdf_offset = output['brdf_offset'][j]
                    renderings.append(img.cpu().detach().numpy())
                    offsets.append(brdf_offset.cpu().detach().numpy())


            imgs_srgb = [np.array(tonemapping('srgb', _img)*255.0).astype(np.uint8) for _img in renderings]
            if not partial:
                save_video(imgs_srgb, fps=30, save_path=f'{output_path}/{name_prefix}_views')
            # Save individual frames as PNGs
            for idx in frames_to_save[name_prefix]:
                save_video([imgs_srgb[idx]], fps=30, save_path=f'{output_path}/{name_prefix}_views_{view_ids[idx]}')

            error = None
            if gts is not None:
                gts = [gts[v] for v in view_ids]
                error = compute_batch_metrics(renderings, gts, lpips_fn=lpips_fn)
                gts_uint8 = [((gt.cpu().detach().numpy())*255.0).astype(np.uint8) for gt in gts]
                # Save video of all ground truth frames
                if not partial:
                    save_video(gts_uint8, fps=30, save_path=f'{output_path}/gt_{name_prefix}')
                # Save individual ground truth frames as PNGs
                for idx in frames_to_save[name_prefix]:
                    save_video([gts_uint8[idx]], fps=30, save_path=f'{output_path}/gt_{name_prefix}_{view_ids[idx]}')



            def blend_offset_with_image(shift, img, channel):
                a = Image.fromarray((colormap_diverging(shift[...,channel], -0.01, 0.01)*255.0).astype(np.uint8))
                return np.asarray(a)

            offset_shift_x = [blend_offset_with_image(val, img, 0) for val, img in zip(offsets, imgs_srgb)]
            offset_shift_y = [blend_offset_with_image(val, img, 1) for val, img in zip(offsets, imgs_srgb)]
            
            # Save videos of all offset frames
            if not partial:
                save_video(offset_shift_x, fps=30, save_path=f'{output_path}/{name_prefix}_offset_shift_x')
                save_video(offset_shift_y, fps=30, save_path=f'{output_path}/{name_prefix}_offset_shift_y')
            for idx in frames_to_save[name_prefix]:
                save_video([offset_shift_x[idx]], fps=30, save_path=f'{output_path}/{name_prefix}_offset_shift_x_{view_ids[idx]}')
                save_video([offset_shift_y[idx]], fps=30, save_path=f'{output_path}/{name_prefix}_offset_shift_y_{view_ids[idx]}')
            
        torch.cuda.empty_cache()
        if not partial:
            # save EXR latents
            os.makedirs(f'{output_path}/latents', exist_ok=True)
            latent_div = range(math.ceil(cfg.latent.channels/3))
            get_latents = lambda x,i: x.detach().cpu().numpy()[3*i:3*i+3].transpose(1,2,0)
            offset_latents_raw = []
            latents_raw = [(f'latent_{3*i}-{min(3*i+3, cfg.latent.channels)}', get_latents(latents[0], i)) for i in latent_div]
            if cfg.use_offset:
                offset_latents_raw = [(f'offset_latent_{3*i}-{min(3*i+3, cfg.latent.channels)}', get_latents(offset_latents[0], i)) for i in latent_div]
            [pyexr.write(f'{output_path}/latents/{name}.exr', _lat.astype(np.float16)) for name, _lat in latents_raw]   
            if cfg.use_offset:
                [pyexr.write(f'{output_path}/latents/{name}.exr', _lat.astype(np.float16)) for name, _lat in offset_latents_raw]

        # output_path = f'{output_path}/../'

        # if accelerator.is_main_process and save_checkpoint:
        #     logger.info(f"Saving model to checkpoint {output_path}/../model.pth")
        #     torch.save(neural_brdf.state_dict(), f'{output_path}/../model.pth')
        #     logger.info(f"Saving latent model to checkpoint {output_path}/../latent_model.pth")
        #     torch.save(latent_module.state_dict(), f'{output_path}/../latent_model.pth')

        # All processes save their error JSONs for this material
        if error is not None:
            error_dir = f'{out_folder}/errors'
            os.makedirs(error_dir, exist_ok=True)
            # error is {view_idx: {'mse': ..., 'psnr': ..., 'mape': ..., 'lpips': ...}}
            # Convert keys to strings for JSON serialization
            error_dict = {str(k): v for k, v in error.items()}
            error_file_path = f'{error_dir}/{accelerator.process_index}_{mat_name}_{split}_{traj}.json'
            with open(error_file_path, 'w') as f:
                json.dump(error_dict, f)
                f.flush()
                os.fsync(f.fileno())

    # Sync all processes after saving all error JSONs
    accelerator.wait_for_everyone()

    if metrics:
        os.makedirs(f'{output_path}/../metrics', exist_ok=True)
        if accelerator.is_main_process:
            for key, yname, logscale in zip(["losses", "sigmas", "lambdas", "lrs"], ["Loss", "Sigma", "Lambda", "Learning Rate"], [True, False, False, False]):
                if key not in metrics:
                    continue
                metric = np.asarray(metrics[key])
                plot_fig = plot_loss(metric, list(range(len(metric))), resolution=(800,800), ylabel=yname, xlabel = 'Steps', legend = yname, color = 'blue', font_size = None, linewidth = 2, y_log_scale = logscale)
                plot_fig.save(f'{output_path}/../metrics/{key}.png')
        

    neural_brdf.train()
    latent_module.train()
    accelerator.wait_for_everyone()

    # Main process collects and aggregates all errors
    if accelerator.is_main_process:
        error_dir = f'{out_folder}/errors'
        run_root = os.path.dirname(out_folder)
        collect_and_update_error_history(error_dir, run_root, out_folder, current_step)


    accelerator.wait_for_everyone()









def render_torch(states, frame_ind, neural_brdf, resolution, device, seed, spp = 4, jitter = True, convert_to_point_light=True, latent_texture=None, offset_texture=None):
    if isinstance(frame_ind, int):
        frame_ind = [frame_ind]
    
    assert len(states) == len(frame_ind), (len(states), len(frame_ind)) # both should be the same length as the batch size


    to_world = torch.from_numpy(np.array([st[v_id]['sensor.to_world'] for st, v_id in zip(states, frame_ind)]))[..., 0]
    light_position = torch.from_numpy(np.array([np.asarray(st[v_id]['area_light_sphere.to_world'])[:3, -1, 0] for st, v_id in zip(states, frame_ind)]))
    fov = torch.from_numpy(np.array([st[v_id]['sensor.x_fov'] for st, v_id in zip(states, frame_ind)]))
    light_intensity = torch.from_numpy(np.array([np.asarray(st[v_id]['area_light_sphere.emitter.radiance.value']) for st, v_id in zip(states, frame_ind)]))

    if light_intensity.ndim == 3: # TODO: remove this in the future
        light_intensity = light_intensity[...,0]
    if light_intensity.ndim == 1:
        light_intensity = light_intensity[None]

    if convert_to_point_light:
        light_intensity = light_intensity * (0.25**2) * np.pi
    
    sensor = {
        "fov": fov,  # allow overriding via fov
        "resolution": (resolution, resolution),
        "to_world": to_world,
    }
    camera = create_camera_from_mitsuba(sensor, device=device)

    plane_width = 4.0
    plane_height = 4.0
    geometry = Plane.create(width=plane_width, height=plane_height, device=device)

    outputs = render(
            camera=camera,
            plane=geometry,
            brdf=neural_brdf,
            image_size=(resolution, resolution),
            light_position=torch.tensor(light_position.tolist(), dtype=torch.float32, device=device),
            light_intensity=torch.tensor(light_intensity.tolist(), dtype=torch.float32, device=device),
            spp=spp,
            jitter=jitter,
            seed=seed,
            latent_texture=latent_texture,
            offset_texture=offset_texture,
    )

    # if single_frame:
    #     outputs = {k: v[0] for k, v in outputs.items()}

    return outputs






def load_scene_states(scene_state_path, n_frames):
    states = [np.load(f'{scene_state_path}/{n_frames} frames/btf_frame__{i:04d}.npy', allow_pickle=True).item() for i in range(n_frames)]
    return states






# obsolete: not used anymore after switching to our own dataset in data.py

