
from json import load
import sys
from typing import Any
import hydra
from utils import plot_loss
sys.path.append('./')
import mitsuba as mi
import drjit as dr
import numpy as np
import torch
import torchvision
from src.utils import save_video
mi.set_variant('cuda_ad_rgb')

from render import *
from src.misc.neumip_mitsuba import NeuralBSDF, load_latents, MitsubaTorchWrapperMLP
from src.misc.neumip_coop import NeuMIPCoop, load_checkpoint
import os
from PIL import Image
from datetime import datetime
from src.utils import make_grid
import math
from tqdm import tqdm
from torchvision.transforms.functional import gaussian_blur
from omegaconf import DictConfig, OmegaConf
from src.utils import load_scene_state, colormap_diverging, unpack_video_to_numpy
import tempfile, base64, os
import numpy as np
import imageio
from IPython.display import HTML
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import cv2
import tempfile, base64, os
import numpy as np
import imageio
from IPython.display import HTML
import mitsuba as mi
import cv2
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageOps
import matplotlib.pyplot as plt
from io import BytesIO
from matplotlib.lines import Line2D
import drjit as dr
from copy import deepcopy
from src.data import get_material_list
from pathlib import Path

def remove_network_prefix(state_dict):
    """
    Strips the 'module.' prefix from keys in a state_dict
    saved from a DistributedDataParallel (DDP) model.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('.network', '')
        new_state_dict[new_k] = v
    return new_state_dict

def tonemapping(tonemapping_type, img):
    def _linear_to_srgb_gpu(x, clamp = True):
        a = 0.055
        if clamp:
            x_pos = dr.maximum(x, 0.0)
        else:
            x_pos = x
        y = dr.select(
            x_pos <= 0.0031308,
            12.92 * x_pos,
            (1 + a) * (x_pos ** (1 / 2.4)) - a  
        )
        # print(dr.min(y), dr.max(y))
        if clamp:
            return dr.clamp(y, 0.0, 1.0)
        else:
            return y
    if tonemapping_type == 'srgb':
        return _linear_to_srgb_gpu(img)
    elif tonemapping_type == 'srgb_no_clamp':
        return _linear_to_srgb_gpu(img, clamp = False)
    elif tonemapping_type == 'reinhard':
        coeff = 7.0
        return coeff * img / (coeff * img + 1)
    elif tonemapping_type == 'none':
        return img
    else:
        raise ValueError(f"Invalid tonemapping: {tonemapping_type}")


@hydra.main(version_base=None, config_path="config", config_name="test")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve = True))
    print(f"Working directory : {os.getcwd()}")
    output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory  : @@@{output_path}@@@")
    
    mi.register_bsdf("neural", lambda props: NeuralBSDF(props))
    neural_brdf = mi.load_dict({'type': 'neural'})

    material_names = []


    material_names = get_material_list(**cfg.materials_info)



    def render_video(cfg, mat_name, output_path):
        output_path = f'{output_path}/{mat_name}'
        os.makedirs(output_path, exist_ok=True)

        latent_textures = load_latents(str(Path(cfg.latents_path) / 'final' / f"{cfg.dataset_name}_split" / f"{cfg.traj}_traj" / mat_name / 'latents') , cfg.latent_channels, cfg.uv_scale, cfg.uv_shift, numpy = not cfg.use_coop)
        offset_latent_textures = load_latents(str(Path(cfg.latents_path) / 'final' / f"{cfg.dataset_name}_split" / f"{cfg.traj}_traj" / mat_name / 'latents') , cfg.latent_channels, cfg.uv_scale, cfg.uv_shift, True, numpy = not cfg.use_coop) if cfg.use_offset else None

        
        if cfg.use_coop:
            model = NeuMIPCoop(
                use_offset=cfg.use_offset,
                latent_channels=cfg.latent_channels,
                network_width=cfg.network_width,
                network_depth=cfg.network_depth
            )
            model.uv_scale = cfg.uv_scale
            assert Path(cfg.mlp_path).exists(), f"MLP path {cfg.mlp_path} does not exist"
            load_checkpoint(
                model, 
                cfg.mlp_path
            )
            neural_brdf.latent_textures = latent_textures
            neural_brdf.offset_latent_textures = offset_latent_textures

        else:
            dr.set_flag(dr.JitFlag.LoopRecord, False)
            dr.set_flag(dr.JitFlag.VCallRecord, False)
            dr.set_flag(dr.JitFlag.VCallOptimize, False)

            model = MitsubaTorchWrapperMLP(
                {'use_offset': cfg.use_offset, 'resolution': 1, 'latent_channels': cfg.latent_channels, 'network_width': cfg.network_width, 'network_depth': cfg.network_depth, "uv_scale": cfg.uv_scale, "uv_shift": cfg.uv_shift}
            )
            missing, unexpected = model.network.load_state_dict(torch.load(cfg.mlp_path), strict=False)
            print('missing', missing)
            print('unexpected', unexpected)
            model.network.latents = torch.cat([torch.from_numpy(p) for p in (latent_textures)], dim = -1).cuda().permute(2,0,1)[None]
            print(model.network.latents.shape)
            if cfg.use_offset:
                model.network.offset_latents = torch.cat([torch.from_numpy(p) for p in (offset_latent_textures)], dim = -1).cuda().permute(2,0,1)[None]
            


        neural_brdf.network = model
        neural_brdf.use_coop = cfg.use_coop


        scene_dict = {
            'cloth': {
                'type': 'ply',
                'filename': 'data/scenes/cloth2.ply',
                'face_normals': False,
                'bsdf': {
                    'type': 'diffuse',
                },
                'to_world': mi.ScalarTransform4f.rotate(axis=[1,0,0], angle=90).rotate(axis=[0,1,0], angle=180)
            },
            'envmap': {
                'type': 'envmap',
                'filename': 'data/to_delete/envmaps/forest.exr'
            },
            'sensor': {
                'type': 'perspective',
                'fov': 60,
                'to_world': mi.ScalarTransform4f.look_at(origin=[-3,3,3], target=[0, 0, 0], up=[0, 1, 0]),
                'film': {
                    'type': 'hdrfilm',
                    'width': cfg.resolution[0],
                    'height': cfg.resolution[1],
                    'pixel_format': 'rgba',
                },
                'sampler': {
                    'type': 'ldsampler',
                    'sample_count': 64
                }
            },
            "plane": {
                'type': 'rectangle',
                'to_world': mi.ScalarTransform4f.scale(12),
                'bsdf': {
                    'type': 'principled',
                    'base_color': {'type': 'bitmap', 'filename': 'data/scenes/Concrete048_2K-JPG/Concrete048_2K-JPG_Color.jpg', 'to_uv': mi.ScalarTransform4f.scale(12)},
                    'roughness': {'type': 'bitmap', 'filename': 'data/scenes/Concrete048_2K-JPG/Concrete048_2K-JPG_Roughness.jpg', 'to_uv': mi.ScalarTransform4f.scale(12)},
                    'metallic': {'type': 'bitmap', 'filename': 'data/scenes/Concrete048_2K-JPG/Concrete048_2K-JPG_Metalness.jpg', 'to_uv': mi.ScalarTransform4f.scale(12)},
                    }
                }
        }


        scene_dict['cloth']['bsdf'] = neural_brdf
        scene = mi.load_dict(scene_dict)
        params = mi.traverse(scene)




        integrator = mi.load_dict({'type': 'direct', 'bsdf_samples': 1, 'emitter_samples': 1, 'hide_emitters': True})
    #     integrator = mi.load_dict({'type': 'aov', 'aovs': 'pos:position',
    #     'my_image': {
    #         'type': 'direct',
    #     }
    # })
        states = None
        n_frames = cfg.n_frames

        from collections import defaultdict

        # cam_loc = cfg.cam_loc
        cam_loc = cfg.cam_loc
        init_lookat = cfg.init_lookat

        lookat_r = cfg.lookat_r
        d= np.array(cam_loc) - np.array(init_lookat)
        r= np.linalg.norm(d)
        d = d/r
        assert lookat_r < r
        lookat = np.array(init_lookat) + (d*lookat_r)
        cam_loc = np.array(init_lookat) + d * cfg.init_r
        up = cfg.up
        # fov_max =cfg.fov_max
        fov_min = cfg.fov_min
        envmap_rot = cfg.envmap_rot
        envmap_rot_range = cfg.envmap_rot_range
        phase_frames = cfg.phase_frames
        cam_degree_rot=cfg.cam_degree_rot
        zoomed_out = cfg.zoomed_out
        cam_degree_rot_zoomout = 20
        # cam_degree_rot_lookat = 35
        cam_tr = lambda o, t, u: mi.Transform4f().look_at(origin=o, target=t, up=u).matrix.numpy().tolist()
        envmap_tr = lambda a: mi.ScalarTransform4f().rotate(axis=[0,0,1], angle=a).matrix.numpy().tolist()



        states = [{"cam_loc": cam_loc, "lookat": lookat, "up": up, "fov":fov_min, "envmap_rot": envmap_rot}]
        print(states[0])


        from copy import deepcopy

        def light_rotation(n, rot_angle):
            state= states[-1]
            init = deepcopy(state)
            for i in range(n):
                state = deepcopy(state)
                t = i/(n-1)
                state["envmap_rot"] = init["envmap_rot"] + t*rot_angle
                states.append(state)

        
        def zoom(n, r_target):
            state= states[-1]
            init = deepcopy(state)
            d = np.array(init["cam_loc"]) - np.array(init_lookat)
            r = np.linalg.norm(d)
            d = d / r
            theta = np.arccos(d[1])
            phi = np.arctan2(d[2], d[0])
            for i in range(n):
                state = deepcopy(state)
                t = i/(n-1)
                new_loc = np.array([np.sin(theta)*np.cos(phi), np.cos(theta), np.sin(theta)*np.sin(phi)])
                state["cam_loc"] = new_loc * (r_target*t + r*(1-t))
                states.append(state)

        def camera_rotation_origin(n, cam_degree_rot):
            state= states[-1]
            init = deepcopy(state)

            d = np.array(init["cam_loc"]) - np.array(init_lookat)
            r = np.linalg.norm(d)
            d = d / r
            theta = np.arccos(d[2])
            phi = np.arctan2(d[1], d[0])

            for i in range(n):
                state = deepcopy(state)
                t = i/(n-1)
                this_phi = phi + t*2*np.pi*cam_degree_rot/360
                new_loc = np.array([np.sin(theta)*np.cos(this_phi), np.sin(theta)*np.sin(this_phi), np.cos(theta)])
                state["cam_loc"] = new_loc * r
                states.append(state)
            

        def camera_rotation_lookat(n, cam_degree_rot):
            state= states[-1]
            init = deepcopy(state)
            print('rotation lookat start')

            print(init['lookat'])

            d = np.array(init["lookat"]) - np.array(init_lookat)
            r = np.linalg.norm(d)
            d = d / r
            theta = np.arccos(d[1])
            phi = np.arctan2(d[2], d[0])

            for i in range(n):
                state = deepcopy(state)
                t = i/(n-1)
                this_phi = phi + t*2*np.pi*cam_degree_rot/360
                new_loc = np.array([np.sin(theta)*np.cos(this_phi), np.cos(theta), np.sin(theta)*np.sin(this_phi)])
                state['lookat'] = new_loc*r
                states.append(state)
            print(state['lookat'])

        # states[-1]['fov'] = fov_max
        key_states = [states[-1]]



        key_states.append(states[-1])
        if zoomed_out:
            light_rotation(phase_frames[0], envmap_rot_range)
            key_states.append(states[-1])
            light_rotation(phase_frames[0], -envmap_rot_range)

            states[-1]['lookat'] = init_lookat
            camera_rotation_origin(phase_frames[1], cam_degree_rot_zoomout)
            key_states.append(states[-1])
            camera_rotation_origin(phase_frames[1], -cam_degree_rot_zoomout)
            # key_states.append(states[-1])
            camera_rotation_origin(phase_frames[1], -cam_degree_rot_zoomout)
            key_states.append(states[-1])
            camera_rotation_origin(phase_frames[1], cam_degree_rot_zoomout)
            states[-1]['lookat'] = lookat
            zoom(phase_frames[2], cfg.mid_zoom_r)
            cuttoff_index = len(states)

        key_states.append(states[-1])
        # states[-1]['fov'] = fov_min
        # states = [states[-1]]
        light_rotation(phase_frames[3], envmap_rot_range)
        key_states.append(states[-1])
        light_rotation(phase_frames[3], -envmap_rot_range)
        # saved_lookat = states[-1]['lookat']
        # states[-1]['lookat'] = init_lookat
        # key_states.append(states[-1])
        camera_rotation_origin(phase_frames[4], cam_degree_rot)
        key_states.append(states[-1])
        camera_rotation_origin(phase_frames[4], -cam_degree_rot)
        # key_states.append(states[-1])
        camera_rotation_origin(phase_frames[4], -cam_degree_rot)
        key_states.append(states[-1])
        camera_rotation_origin(phase_frames[4], cam_degree_rot)
        # key_states.append(states[-1])
        # states[-1]['lookat'] = saved_lookat

        # zoom(phase_frames[2], 1.5)
        # key_states.append(states[-1])
        # camera_rotation_origin(phase_frames[3], cam_degree_rot_zoomed)
        # key_states.append(states[-1])    
        # camera_rotation_origin(phase_frames[3], -cam_degree_rot_zoomed)
        # camera_rotation_origin(phase_frames[3], -cam_degree_rot_zoomed)
        # key_states.append(states[-1])
        # camera_rotation_origin(phase_frames[3], cam_degree_rot_zoomed)

        # camera_rotation_lookat(phase_frames[3], cam_degree_rot_zoomed_lookat)
        # camera_rotation_lookat(phase_frames[3], -cam_degree_rot_lookat)
        # camera_rotation_lookat(phase_frames[3], -cam_degree_rot_lookat)
        # camera_rotation_lookat(phase_frames[3], cam_degree_rot_lookat)
        # zoom(phase_frames[4], r)



        def load_states(state, params):
            params['sensor.to_world'] = cam_tr(state['cam_loc'], state['lookat'], state['up'])
            params['envmap.to_world'] = envmap_tr(state['envmap_rot'])
            params['sensor.x_fov'] = state['fov']
            params.update()

        torch.set_grad_enabled(False)
        dr.disable_grad(params)





        tasks = [cfg.video, cfg.key_frames]
        task_states = [states, key_states]
        task_is_video = [True, False]

        for task, states, is_video in zip(tasks, task_states, task_is_video):
            if not task:
                continue
            torch.cuda.empty_cache()
            renderings = []
            n_frames = len(states)
            for i in tqdm(range(n_frames)):
                # if cfg.frame is not None:
                #     if i != cfg.frame:
                #         continue
                load_states(states[i], params)
                
                img = mi.render(scene, integrator = integrator, params = params, spp=cfg.spp)
                dr.flush_malloc_cache()
                dr.flush_kernel_cache()
                # mi.util.write_bitmap(f'{output_path}/img_{i:04d}.exr', img[:,:,4:])
                # img = (img.numpy()[:,:,4:]*255).astype(np.uint8)
                img_tmapped = np.array(mi.Bitmap(img.numpy()).convert(mi.Bitmap.PixelFormat.RGBA, mi.Struct.Type.UInt8, srgb_gamma=True))
                renderings.append(img_tmapped)
            if is_video:
                save_video(renderings, fps=30, save_path=f'{output_path}/video')
                if zoomed_out:
                    save_video(renderings[cuttoff_index:], fps=30, save_path=f'{output_path}/video_zoomed_in')
            else:
                for i in range(len(renderings)):
                    save_video([renderings[i]], save_path=f'{output_path}/image_{i:04d}')


        if cfg.offset_video and cfg.use_offset:
            print('Rendering offset ...')
            class OffsetIntegrator(mi.SamplingIntegrator):
                def sample(self: mi.SamplingIntegrator, scene: mi.Scene, sampler, ray: mi.RayDifferential3f, medium: mi.Medium = None, active: bool = True):
                    si:mi.SurfaceInteraction3f = scene.ray_intersect(ray, mi.RayFlags.All, True, active)
                    valid_ray = active & si.is_valid()
                    # si.wi is passed instead of wo, which is fine because we will factor out the cosine, and also it does not have an effect on the offset value
                    offset_value = si.bsdf(ray).eval(mi.BSDFContext(), si, si.wi, active)
                    valid_ray&=(si.bsdf() == neural_brdf)
                    result = dr.select(valid_ray, mi.Vector3f(offset_value), mi.Vector3f(0))
                    return result, valid_ray, []
            
            if cfg.use_coop:
                neural_brdf.network.return_offset = True
            else:
                neural_brdf.network.network.return_offset = True
                
            scene_offset_only = mi.load_dict(scene_dict)
            params_offset_only = mi.traverse(scene_offset_only)
            def render_offset(states, name):
                torch.cuda.empty_cache()
                offset_renderings = []
                for i in range(len(states)):
                    load_states(states[i], params_offset_only)
                    img = mi.render(scene_offset_only, integrator = OffsetIntegrator(mi.Properties()), params = params_offset_only, spp=4)
                    dr.flush_malloc_cache()
                    dr.flush_kernel_cache()
                    offset_renderings.append(img.numpy())

                
                offset_shift_x = [colormap(val, 0) for val in offset_renderings]
                offset_shift_y = [colormap(val, 1) for val in offset_renderings]
                return offset_shift_x, offset_shift_y
                
            
            for task, states, is_video in zip(tasks, task_states, task_is_video):
                if not task:
                    continue
                offset_shift_x, offset_shift_y = render_offset(states, 'video' if is_video else 'image')
                name = 'video' if is_video else 'image'
                if is_video:
                    save_video(offset_shift_x, fps=24, save_path=f'{output_path}/{name}_shift_x')
                    save_video(offset_shift_y, fps=24, save_path=f'{output_path}/{name}_shift_y')
                    if zoomed_out:
                        save_video(offset_shift_x[cuttoff_index:], fps=24, save_path=f'{output_path}/{name}_shift_x_zoomed_in')
                        save_video(offset_shift_y[cuttoff_index:], fps=24, save_path=f'{output_path}/{name}_shift_y_zoomed_in')
                else:
                    for i in range(len(offset_shift_x)):
                        save_video([offset_shift_x[i]], save_path=f'{output_path}/{name}_shift_x_{i:04d}')
                        save_video([offset_shift_y[i]], save_path=f'{output_path}/{name}_shift_y_{i:04d}')

    for mat_name in material_names:

        render_video(cfg, mat_name, output_path)

def colormap(shift, channel):
    alpha = 255*(shift[...,3:4] + 0)
    alpha = alpha.astype(np.uint8)
    a = Image.fromarray((colormap_diverging(shift[...,channel], -0.005, 0.005)*255.0).astype(np.uint8))
    result = np.asarray(a)
    result = np.concatenate([result, alpha], axis=-1)
    return result

if __name__ == '__main__':
    main()
