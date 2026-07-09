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
from datetime import datetime
from src.utils import make_grid
import math
from tqdm import tqdm
from torchvision.transforms.functional import gaussian_blur
from omegaconf import DictConfig, OmegaConf
from src.utils import show_triplet_video
from src.utils import show_video_from_mitsuba_frames, load_scene_state, colormap_diverging, unpack_video_to_numpy
from src.utils import to_pil
from common import seed_everything, tonemapping, calc_loss, render_torch, validation_cameras, load_checkpoint, load_scene_states, final_validation
from src.lrm import load_LRM, eval_lrm
from diffsynth.trainers.unified_dataset import UnifiedDataset
import math
from torch.utils.data import DataLoader



torch.cuda.set_per_process_memory_fraction(0.9, device=0)



@hydra.main(version_base=None, config_path="config", config_name="inverse_render")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve = True))
    print(f"Working directory : {os.getcwd()}")
    output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory  : @@@{output_path}@@@")

    cfg.use_LRM=True

    
    renderer = StreamingBTFRenderer(resolution=cfg.resolution, grid_size=cfg.mesh_grid_size, only_center_mesh_instance=cfg.only_center_mesh_instance, arealight=cfg.use_area_light)
    
    

    val_states = validation_cameras(lightgeight = renderer.lightgeight, area_light_radius = renderer.area_light_radius, camera_height = renderer.camera_height, base_light_intensity = renderer.base_light_intensity, camera_radius = renderer.camera_radius, use_area_light=True)


    if cfg.use_LRM:
        LRM = load_LRM()


    device = torch.device('cuda')

    if cfg.seed is not None:
        seed_everything(cfg.seed)

    neural_brdf = load_network({"use_offset": cfg.use_offset, "resolution": cfg.resolution, "latent_channels": cfg.latent_channels, "network_width": cfg.network_width, "network_depth": cfg.network_depth, "standalone_latents": False}, device=device)
    # neural_brdf.load_state_dict(torch.load('/home/shadadan/NeuGen/outputs/second experiments/matsynth_offset/10-19-07-Wood Grid 01-L2rel-offset_True-tonemapp_srgb_no_clamp-netsize_l12-32x2/final/model.pth', ),strict=False)

    if cfg.use_checkpoint:
        neural_brdf = load_checkpoint(cfg)
        raise NotImplementedError('Checkpoint loading is not implemented for LRM')



    def validation(cfg, neural_brdf, output_path, train_gts, train_gts_raw, val_gts, val_gts_raw, loss_values, sigma_values, val_data):

        with torch.no_grad():
            lrm_output = eval_lrm(LRM, val_data).float()
            latents, offset_latents = lrm_output.chunk(2, dim = 1)
            neural_brdf.latents = latents
            if neural_brdf.use_offset:
                neural_brdf.offset_latents = offset_latents
            train_gts = [torch.from_numpy(np.asarray(frame).astype(np.float32)/255.0).to(device) for frame in val_data['video']]

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
        latents = [(f'latent_{3*i}-{min(3*i+3, cfg.latent_channels)}', get_latents(neural_brdf.latents, i)) for i in latent_div]
        if neural_brdf.use_offset:
            offset_latents = [(f'offset_latent_{3*i}-{min(3*i+3, cfg.latent_channels)}', get_latents(neural_brdf.offset_latents, i)) for i in latent_div]


        if save_exrs:
            [mi.util.write_bitmap(f'{output_path}/{name}.exr', mi.Bitmap(_lat)) for name, _lat in latents]   
            if neural_brdf.use_offset:
                [mi.util.write_bitmap(f'{output_path}/{name}.exr', mi.Bitmap(_lat)) for name, _lat in offset_latents]

            

        torch.save(neural_brdf.state_dict(), f'{output_path}/model.pth')

        loss_fig = plot_loss(np.asarray(loss_values), list(range(len(loss_values))), resolution=(800,800), ylabel='Loss', xlabel = 'Epochs', legend = 'Loss', color = 'blue', font_size = None, linewidth = 2)
        sigmas_fig = plot_loss(np.asarray(sigma_values), list(range(len(sigma_values))), resolution=(800,800), ylabel='Sigma', xlabel = 'Steps', legend = 'Sigma', color = 'blue', font_size = None, linewidth = 2, y_log_scale = False)
        
        loss_fig.save(f'{output_path}/loss.png')
        sigmas_fig.save(f'{output_path}/sigmas.png')
        


    n_frames = cfg.n_frames = 129
    n_frames = 129
    dataset_base_path = '/mnt/c/Users/shadadan/Downloads/btf_dataset_output7/btf_dataset_output7'
    dataset_metadata_path = '/mnt/c/Users/shadadan/Downloads/btf_dataset_output7/btf_dataset_output7/metadata_clean.csv'

    dataset_repeat = 1
    data_file_keys = 'video'
    max_pixels = cfg.resolution**2
    height = cfg.resolution
    width = cfg.resolution
    num_frames = cfg.n_frames
    dataset = UnifiedDataset(
        base_path=dataset_base_path,
        metadata_path=dataset_metadata_path,
        repeat=dataset_repeat,
        data_file_keys=data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=dataset_base_path,
            max_pixels=max_pixels,
            height=height,
            width=width,
            num_frames=num_frames,
        )
        )
    class _SafeDataset(torch.utils.data.Dataset):
        def __init__(self, inner_dataset):
            self.inner_dataset = inner_dataset
        def __len__(self):
            return len(self.inner_dataset)
        def __getitem__(self, index):

            if index < 1800:
                return None
        
            try:
                return self.inner_dataset[index]
            except Exception as exc:
                print(f"[WARN] Skipping sample {index} due to read error: {exc}")
                return None

    def _collate_skip_none(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        return batch[0]

    dataloader = DataLoader(_SafeDataset(dataset), shuffle=False, collate_fn=_collate_skip_none, num_workers=8)


        


    val_data = dataset[3688] 


    states = load_scene_states(cfg.scene_state_path, n_frames)
    gts_raw, val_gts, val_gts_raw = None, None, None


    if cfg.optimize:
        # load scene state and ground truth
        
        LRM.freeze_except(['dit'])
        torch_optim = torch.optim.AdamW(
            [{"params": neural_brdf.parameters(), "lr": 0.001},
                {"params": LRM.dit.parameters(), "lr": 0.00005}], lr = cfg.lr)


        loss_values, sigma_values = [], []

        # pbar = tqdm(total=len(dataloader))
        pbar = tqdm(total=15000)

        gt_latent = torch.load('/home/shadadan/NeuGen/outputs/second experiments/matsynth_offset/10-19-07-Wood Grid 01-L2rel-offset_True-tonemapp_srgb_no_clamp-netsize_l12-32x2/final/model.pth')
        gt_latent = torch.cat([gt_latent['latents'], gt_latent['offset_latents']], dim = 1).to(device)
        gt_latent = torch.nn.functional.interpolate(gt_latent, size=(cfg.resolution, cfg.resolution), mode='bilinear')
        data = val_data

        gts = [torch.from_numpy(np.asarray(frame).astype(np.float32)/255.0).to(device) for frame in data['video']]

        gts = torch.stack([gts[i] for i in range(0,len(gts), 8)], dim=0)
        this_states = [states[i] for i in range(0,len(states), 8)]
        data['video'] = [data['video'][i] for i in range(0,len(data['video']), 8)]


        # for step, data in enumerate(dataloader):
        for step in range((15000)):

            if data is None:
                continue
            # gts = [torch.from_numpy(np.asarray(frame).astype(np.float32)/255.0).to(device) for frame in data['video']]

            # gts = torch.stack([gts[i] for i in range(0,len(gts), 8)], dim=0)
            # this_states = [states[i] for i in range(0,len(states), 8)]
            # data['video'] = [data['video'][i] for i in range(0,len(data['video']), 8)]


            print(len(data['video']), len(gts), len(this_states))


            if (step) % 500 == 0:
                os.makedirs(f'{output_path}/step_{step}', exist_ok=True)
                validation(cfg, neural_brdf, f'{output_path}/step_{step}', gts, gts_raw, val_gts, val_gts_raw, loss_values, sigma_values, val_data)

                os.makedirs(f'{output_path}/dumps/step_{step}', exist_ok=True)
                for i in range(len(data['video'])):                    
                    to_pil(gts[i]).save(f'{output_path}/dumps/step_{step}/gt_{i}.png')
                    data['video'][i].save(f'{output_path}/dumps/step_{step}/video_{i}.png')


            lrm_output_og = eval_lrm(LRM, data)

            if step % 500 == 0:

                mi.Bitmap(mi.TensorXf(lrm_output_og[0, 0:3].permute(1, 2, 0).detach())).write(f'{output_path}/{step}_latents0-3.exr')
                mi.Bitmap(mi.TensorXf(gt_latent[0, 0:3].permute(1, 2, 0).detach())).write(f'{output_path}/gt_latents0-3.exr')



            print(lrm_output_og.shape, gt_latent.shape)

            # loss = torch.mean((lrm_output_og - gt_latent)**2)
            # loss.backward()
            

            torch.cuda.empty_cache()

            # lrm_output = lrm_output_og.detach().clone().requires_grad_(True)
            latents, offset_latents = lrm_output_og.chunk(2, dim = 1)
            neural_brdf.latents = latents

            if neural_brdf.use_offset:                    
                neural_brdf.offset_latents = offset_latents


            # if (step) % 1000 == 0:
            #         mi.Bitmap(mi.TensorXf(latents[0, 0:3].permute(1, 2, 0).detach())).write(f'{output_path}/dumps/step_{step}/latents.exr')



            n_frames_this = len(this_states)

            losses = 0

            b = 17

            outputs = render_torch(
                    states=this_states, 
                    frame_ind=[0, len(this_states)], 
                    neural_brdf=neural_brdf, 
                    resolution=cfg.resolution, 
                    device=device, 
                    seed=step,
                    convert_to_point_light=True
                )


            img_raw = outputs['image']
            vis_mask = (outputs['visibility'] > 0.999999)[...,0].detach() # shape = H,w

            img = tonemapping(cfg.tonemapped, img_raw)

            if img_raw.shape[0] == 0:
                print('BUG: img is empty', gts.shape, len(data['video']), img.shape, img_raw.shape)
                continue



            loss = torch.mean(((img - gts)**2)[vis_mask])


            loss, loss_pp = calc_loss(img, gts, cfg.loss_func, vis_mask, None)
            loss.backward()
            # losses += loss.detach().clone().item()/inner_iters





            # lrm_output_og.backward(lrm_output.grad.detach().clone()/inner_iters)
            

            torch_optim.step()
            torch_optim.zero_grad()


            # del grad_lrm_out, lrm_output_og, lrm_output, latents, offset_latents


            # torch.cuda.empty_cache()
            # torch.cuda.empty_cache()
            # gc.collect()


            loss_values.append(loss.detach().item())
            pbar.set_postfix(loss=f"{loss_values[-1]:.4f}")
            pbar.update(1)

        os.makedirs(f'{output_path}/final', exist_ok=True)
        validation(cfg, neural_brdf, f'{output_path}/final', gts, gts_raw, val_gts, val_gts_raw, loss_values, sigma_values, val_data)
        np.save(f'{output_path}/loss_values.npy', loss_values)



    final_validation(cfg, neural_brdf, output_path, gts, states, val_states, n_frames, device)



if __name__ == '__main__':
    main()





