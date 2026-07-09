import sys
from typing import Any
import hydra

from utils import plot_loss, show_video_from_mitsuba_frames
sys.path.append('/home/shadadan/NeuGen')
import mitsuba as mi
import drjit as dr
import numpy as np
import torch
import torchvision
mi.set_variant('cuda_ad_rgb')
from render import *
from src.neumip_nets import load_network
import os
from PIL import Image
import math
from tqdm import tqdm
from torchvision.transforms.functional import gaussian_blur
from omegaconf import DictConfig, OmegaConf
from src.utils import colormap_diverging
from src.utils import to_pil
from common import seed_everything, tonemapping, calc_loss, validation_cameras, final_validation, render_torch, load_gts, load_checkpoint, load_scene_states, load_prerendered_matsynth
import math
from src.data import load_dataset



torch.cuda.set_per_process_memory_fraction(0.9, device=0)


@hydra.main(version_base=None, config_path="config", config_name="inverse_render")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve = True))
    print(f"Working directory : {os.getcwd()}")
    output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory  : @@@{output_path}@@@")




    device = torch.device('cuda')

    if cfg.seed is not None:
        seed_everything(cfg.seed)

    # dataset, dataloader = load_prerendered_matsynth(cfg.n_frames, cfg.resolution, 1, return_dataloader = True, subset = [3688, 1861 - 2, 2550 - 2, 3162 - 2, 1488 - 2, 350 - 2, 2987 - 2, 1670 - 2, 3112 - 2, 5289 - 2])
    dataset = load_dataset(
        dataset_path='data/matsynth77/train', 
        input_type='video', 
        tonemapping_type=cfg.tonemapped, 
        resolution=cfg.resolution, 
        materials_info={'csv_path':'data/matsynth77/train/metadata_clean.csv', 'start':0, 'end':None, 'every_other':1, 'specific_list':[3688]}, 
        frames_info={'start':0, 'end':cfg.n_frames, 'every_other':1}, 
        dataset_n_frames=cfg.n_frames, 
        lazy_load=False)

    
    states = load_scene_states(Path(cfg.scene_state_path)/'train', cfg.n_frames)
    states = [states[i] for i in range(0,cfg.n_frames, 1)]
    val_states = load_scene_states(Path(cfg.scene_state_path)/'test', 320)






    print('dataset length: ',len(dataset), dataset.n_mats, ' materials')
    neural_brdf = load_network({"use_offset": cfg.use_offset, "resolution": cfg.resolution, "latent_channels": cfg.latent_channels, "network_width": cfg.network_width, "network_depth": cfg.network_depth, "standalone_latents": True, "num_materials": dataset.n_mats}, device=device)
    print('neural_brdf loaded')
    # neural_brdf.load_state_dict(torch.load('/home/shadadan/NeuGen/outputs/2025-11-10/14-34-17-all-L2-offset_True-tonemapp_srgb_no_clamp-netsize_l12-64x4/final/model.pth'), strict=False)



    if cfg.use_checkpoint:
        neural_brdf = load_checkpoint(cfg)






    def validation(cfg, neural_brdf, output_path, train_gts, train_gts_raw, val_gts, val_gts_raw, loss_values, sigma_values, mat_id = 0):
        os.makedirs(f'{output_path}', exist_ok=True)

        for state_list, frame_ind, name_prefix, gts, gts_raw in zip([states, val_states], [0, 200], ['train', 'val'], [train_gts, val_gts], [train_gts_raw, val_gts_raw]):
            save_exrs = 'final' in output_path

            with torch.no_grad():
                output = render_torch(states=state_list, frame_ind=frame_ind, neural_brdf=neural_brdf, resolution=cfg.resolution, device=device, seed=0, spp=4)
                img_raw = output['image']
                brdf_offset = output['brdf_offset']
                img = tonemapping(cfg.tonemapped, img_raw)

            if save_exrs:
                mi.util.write_bitmap(f'{output_path}/{name_prefix}_fitted_linear.exr', mi.Bitmap(mi.TensorXf(img_raw.clamp(min=0.0, max=1.0))))
                if gts_raw is not None:
                    mi.util.write_bitmap(f'{output_path}/{name_prefix}_gt_linear.exr', mi.Bitmap(mi.TensorXf(gts_raw[frame_ind].clamp(min=0.0, max=1.0))))
                mi.util.write_bitmap(f'{output_path}/{name_prefix}_brdf_offset_x.exr', mi.Bitmap(mi.TensorXf(brdf_offset[...,0])))

            Image.fromarray((colormap_diverging(brdf_offset[...,0].cpu().detach().numpy(), -0.01, 0.01)*255.0).astype(np.uint8)).save(f'{output_path}/{name_prefix}_brdf_offset_x.png')

            img_pil = to_pil(img.clamp(min=0.0, max=1.0))
            img_pil.save(f'{output_path}/{name_prefix}_fitted.png')

            if gts is not None:
                gt = gts[frame_ind]
                gt_pil = to_pil(gt.clamp(min=0.0, max=1.0))
                gt_pil.save(f'{output_path}/{name_prefix}_gt.png')

        latent_div = range(math.ceil(cfg.latent_channels/3))

        get_latents = lambda x,i: x.detach().cpu().numpy()[0,3*i:3*i+3].transpose(1,2,0)
        offset_latents, offset_latents_pil = [], []
        latents = [(f'latent_{3*i}-{min(3*i+3, cfg.latent_channels)}', get_latents(neural_brdf.latents[:, mat_id], i)) for i in latent_div]
        if neural_brdf.use_offset:
            offset_latents = [(f'offset_latent_{3*i}-{min(3*i+3, cfg.latent_channels)}', get_latents(neural_brdf.offset_latents[:, mat_id], i)) for i in latent_div]



        if save_exrs:
            [mi.util.write_bitmap(f'{output_path}/{name}.exr', mi.Bitmap(_lat)) for name, _lat in latents]   
            if neural_brdf.use_offset:
                [mi.util.write_bitmap(f'{output_path}/{name}.exr', mi.Bitmap(_lat)) for name, _lat in offset_latents]

        output_path = f'{output_path}/../'

        if 'final' in output_path:
            torch.save(neural_brdf.state_dict(), f'{output_path}/model.pth')

        loss_fig = plot_loss(np.asarray(loss_values), list(range(len(loss_values))), resolution=(800,800), ylabel='Loss', xlabel = 'Epochs', legend = 'Loss', color = 'blue', font_size = None, linewidth = 2)
        sigmas_fig = plot_loss(np.asarray(sigma_values), list(range(len(sigma_values))), resolution=(800,800), ylabel='Sigma', xlabel = 'Steps', legend = 'Sigma', color = 'blue', font_size = None, linewidth = 2, y_log_scale = False)
        
        loss_fig.save(f'{output_path}/loss.png')
        sigmas_fig.save(f'{output_path}/sigmas.png')
        

    
    


    n_frames = cfg.n_frames

    # states = load_scene_states(cfg.scene_state_path/'train', cfg.n_frames)
    # val_states = load_scene_states(cfg.scene_state_path/'test', cfg.n_frames)


    if cfg.optimize:

        n_iterations = cfg.n_iterations * dataset.n_mats
        torch_optim = torch.optim.Adam(neural_brdf.parameters(), lr = cfg.lr)
        loss_values, sigma_values = [], []
        half_life = n_iterations // 3
        pbar = tqdm(total=n_iterations)

        for step in range(n_iterations+1):

            rand_idx =int(np.random.choice(range(len(dataset))))
            data = dataset[rand_idx]
            gt = data['images'].to(device)
            mat_id = data['mat_ids']
            view_id = data['view_ids']

            sigma = int(math.exp(-step/half_life)*4)*2 + 1
            sigma_values.append(sigma)
            # print('step:', step)


            if (step) % 2000 == 0:
                for mat_id in range(min(10, len(dataset.datasets))):
                    os.makedirs(f'{output_path}/step_{step}/material_{mat_id}', exist_ok=True)
                    neural_brdf.material_id = mat_id
                    videos_val = dataset.datasets[mat_id].input_images.to(device)
                    validation(cfg, neural_brdf, f'{output_path}/step_{step}/material_{mat_id}', videos_val, None, None, None, loss_values, sigma_values, mat_id = mat_id)


            neural_brdf.material_id = mat_id
            gts_raw, val_gts, val_gts_raw = None, None, None



            with torch.no_grad():
                neural_brdf.latents.data[:, mat_id] = gaussian_blur(neural_brdf.latents[:, mat_id], kernel_size= sigma)
                if neural_brdf.use_offset:
                    neural_brdf.offset_latents.data[:, mat_id] = gaussian_blur(neural_brdf.offset_latents[:, mat_id], kernel_size= sigma+2) # bigger sigma for the offset

            
            outputs = render_torch(
                    states=states, 
                    frame_ind=view_id, 
                    neural_brdf=neural_brdf, 
                    resolution=cfg.resolution, 
                    device=device, 
                    seed=step,
                    convert_to_point_light=True
                )



            img_raw = outputs['image']
            vis_mask = (outputs['visibility'] > 0.999999)[...,0].detach() # shape = H,w
            assert vis_mask.ndim == 2

            img = tonemapping(cfg.tonemapped, img_raw)


            loss, _ = calc_loss(img, gt, cfg.loss_func, vis_mask, None)


            if cfg.use_offset and cfg.offset_lambda is not None:
                lambdaa = 1 / (step/cfg.offset_lambda + 1)
                loss+=lambdaa*(outputs['brdf_offset']).abs().mean()

            loss.backward()

            torch_optim.step()
            torch_optim.zero_grad()

            if mat_id == 0:
                loss_values_this = loss.detach().clone().item()
                loss_values.append(loss_values_this)

            pbar.set_postfix(loss=f"{loss.item():.4f}", sigma=sigma, mat_id=mat_id, view_id=view_id)
            pbar.update(1)


        for mat_id in range(min(10, dataset.n_mats)):  
            gts = dataset.datasets[mat_id].input_images
            neural_brdf.material_id = mat_id
            validation(cfg, neural_brdf, f'{output_path}/final/material_{mat_id}', gts, gts_raw, val_gts, val_gts_raw, loss_values, sigma_values)
        np.save(f'{output_path}/loss_values.npy', loss_values)
        
    for mat_i in range(min(10, dataset.n_mats)):
        gts = dataset.datasets[mat_i].input_images
        neural_brdf.material_id = mat_i
        final_validation(cfg, neural_brdf, f'{output_path}/{mat_i}', gts, states, val_states, n_frames, device)








if __name__ == '__main__':
    main()





