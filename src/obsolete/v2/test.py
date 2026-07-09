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



torch.cuda.set_per_process_memory_fraction(0.9, device=0)


@hydra.main(version_base=None, config_path="config", config_name="test")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve = True))
    print(f"Working directory : {os.getcwd()}")
    output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory  : @@@{output_path}@@@")

    
    renderer = StreamingBTFRenderer(resolution=cfg.resolution, grid_size=cfg.mesh_grid_size, only_center_mesh_instance=cfg.only_center_mesh_instance, arealight=cfg.use_area_light)
    val_states = validation_cameras(lightgeight = renderer.lightgeight, area_light_radius = renderer.area_light_radius, camera_height = renderer.camera_height, base_light_intensity = renderer.base_light_intensity, camera_radius = renderer.camera_radius, use_area_light=cfg.use_area_light)


    device = torch.device('cuda')

    if cfg.seed is not None:
        seed_everything(cfg.seed)

    dataset, dataloader = load_prerendered_matsynth(cfg.n_frames, cfg.resolution, 1, return_dataloader = True, subset = [3688, 1861 - 2, 2550 - 2, 3162 - 2, 1488 - 2, 350 - 2, 2987 - 2, 1670 - 2, 3112 - 2, 5289 - 2])

    new_dataset = []
    for i in range(len(dataset)):
        if dataset[i] is not None:
            torch_frames = [torch.from_numpy(np.asarray(f).astype(np.float32)/255.0).to(device) for f in dataset[i]['video']]
            new_dataset.append(torch_frames)

    del dataset, dataloader
    dataset = new_dataset



    print('dataset length:',len(dataset))
    neural_brdf = load_network({"use_offset": cfg.use_offset, "resolution": cfg.resolution, "latent_channels": cfg.latent_channels, "network_width": cfg.network_width, "network_depth": cfg.network_depth, "standalone_latents": True, "num_materials": len(dataset)}, device=device)
    neural_brdf.load_state_dict(torch.load('/home/shadadan/NeuGen/outputs/2025-11-10/14-34-17-all-L2-offset_True-tonemapp_srgb_no_clamp-netsize_l12-64x4/final/model.pth'), strict=False)
    neural_brdf = load_checkpoint(cfg)
    
    

    n_frames = cfg.n_frames

    # gts, gts_raw, val_gts, val_gts_raw, n_frames = load_gts(cfg, device)
    states = load_scene_states(cfg.scene_state_path, n_frames)



    
    for mat_i in range(len(dataset)):
        gts = dataset[mat_i]
        neural_brdf.material_id = mat_i
        os.makedirs(f'{output_path}/{mat_i}', exist_ok=True)
        final_validation(cfg, neural_brdf, f'{output_path}/{mat_i}', gts, states, val_states, n_frames, device)








if __name__ == '__main__':
    main()





