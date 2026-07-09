import sys
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
from src.neumip_mitsuba import NeuralBSDF, load_network, load_latents, NeuralBSDFOffsetOnly
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
from common import seed_everything, tonemapping, calc_loss
from src.lrm import load_LRM, eval_lrm
from diffsynth.trainers.unified_dataset import UnifiedDataset
import math


from src.torch_renderer import (
    Plane,
    Texture,
    create_camera_from_mitsuba,
    DiffuseBRDF,
    geometry,
    render,
)



torch.cuda.set_per_process_memory_fraction(0.9, device=0)

def render_torch(states, frame_ind, geometry, neural_brdf, resolution, device, seed, spp = 4, jitter = True, convert_to_point_light=True):
    if isinstance(frame_ind, int):
        idxs = [frame_ind, frame_ind+1]
    else:
        idxs = frame_ind
    to_world = torch.tensor([states[i]['sensor.to_world'] for i in range(idxs[0], idxs[1])])[..., 0]
    light_position = torch.tensor([np.asarray(states[i]['area_light_sphere.to_world'])[:3, -1, 0] for i in range(idxs[0], idxs[1])])
    fov = torch.tensor([states[i]['sensor.x_fov'] for i in range(idxs[0], idxs[1])])
    light_intensity = torch.tensor([np.asarray(states[i]['area_light_sphere.emitter.radiance.value']) for i in range(idxs[0], idxs[1])])

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
    )

    return outputs






@hydra.main(version_base=None, config_path="config", config_name="inverse_render")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve = True))
    print(f"Working directory : {os.getcwd()}")
    output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory  : @@@{output_path}@@@")

    
    renderer = StreamingBTFRenderer(resolution=cfg.resolution, grid_size=cfg.mesh_grid_size, only_center_mesh_instance=cfg.only_center_mesh_instance, arealight=cfg.use_area_light)
    
    

    val_states = validation_cameras(lightgeight = renderer.lightgeight, area_light_radius = renderer.area_light_radius, camera_height = renderer.camera_height, base_light_intensity = renderer.base_light_intensity, camera_radius = renderer.camera_radius, use_area_light=True)


    if cfg.use_LRM:
        LRM = load_LRM()


    device = torch.device('cuda')

    if cfg.seed is not None:
        seed_everything(cfg.seed)

    neural_brdf = load_network({"use_offset": cfg.use_offset, "resolution": cfg.resolution, "latent_channels": cfg.latent_channels, "network_width": cfg.network_width, "network_depth": cfg.network_depth, "standalone_latents": not cfg.use_LRM}, device=device)
    plane_width = 4.0
    plane_height = 4.0
    geometry = Plane.create(width=plane_width, height=plane_height, device=device)



    if cfg.use_checkpoint:
        print('Loading checkpoint...')
        train_cfg = OmegaConf.load(Path(cfg.checkpoint) / "../../.hydra/config.yaml")
        cfg.network_width = train_cfg.network_width
        cfg.network_depth = train_cfg.network_depth
        cfg.latent_channels = train_cfg.latent_channels
        cfg.use_offset = train_cfg.use_offset
        neural_brdf = load_network({"use_offset": cfg.use_offset, "resolution": cfg.resolution, "latent_channels": cfg.latent_channels, "network_width": cfg.network_width, "network_depth": cfg.network_depth}, device=device)

        

        neural_brdf.load_state_dict(torch.load(cfg.checkpoint))
        print('Checkpoint loaded')







    def validation(cfg, neural_brdf, output_path, train_gts, train_gts_raw, val_gts, val_gts_raw, loss_values, sigma_values):
        for state_list, frame_ind, name_prefix, gts, gts_raw in zip([states, val_states], [0, 200], ['train', 'val'], [train_gts, val_gts], [train_gts_raw, val_gts_raw]):
            save_exrs = 'final' in output_path

            with torch.no_grad():
                output = render_torch(states=state_list, frame_ind=frame_ind, geometry=geometry, neural_brdf=neural_brdf, resolution=cfg.resolution, device=device, seed=0, spp=4)
                img_raw = output['image'][0]
                brdf_offset = output['brdf_offset'][0]
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

        # latents_pil = [(name, to_pil(np.abs(_lat)*10)) for name, _lat in latents]
        # if neural_brdf.use_offset:
        #     offset_latents_pil = [(name, to_pil(np.abs(_lat)*10)) for name, _lat in offset_latents]


        if save_exrs:
            [mi.util.write_bitmap(f'{output_path}/{name}.exr', mi.Bitmap(_lat)) for name, _lat in latents]   
            if neural_brdf.use_offset:
                [mi.util.write_bitmap(f'{output_path}/{name}.exr', mi.Bitmap(_lat)) for name, _lat in offset_latents]

            


        # [mi.util.write_bitmap(f'{output_path}/latent_{3*i}-{3*i+3}.exr', mi.Bitmap(neural_brdf.network.network.latents.detach().cpu().numpy()[0,3*i:3*i+3].transpose(1,2,0))) for i in latent_div]
        # mi.util.write_bitmap(f'{output_path}/fitted_latent_6.exr', mi.Bitmap(neural_brdf.network.network.latents.detach().cpu().numpy()[0,6:].transpose(1,2,0)))
            # mi.util.write_bitmap(f'{output_path}/fitted_offset_latent_6.exr', mi.Bitmap(neural_brdf.network.network.offset_latents.detach().cpu().numpy()[6].transpose(1,2,0)))


        torch.save(neural_brdf.state_dict(), f'{output_path}/model.pth')

        loss_fig = plot_loss(np.asarray(loss_values), list(range(len(loss_values))), resolution=(800,800), ylabel='Loss', xlabel = 'Epochs', legend = 'Loss', color = 'blue', font_size = None, linewidth = 2)
        sigmas_fig = plot_loss(np.asarray(sigma_values), list(range(len(sigma_values))), resolution=(800,800), ylabel='Sigma', xlabel = 'Steps', legend = 'Sigma', color = 'blue', font_size = None, linewidth = 2, y_log_scale = False)
        
        # grid_imgs = [gt_pil, img_pil] + [latent_pil for _, latent_pil in latents_pil] + [offset_latent_pil for _, offset_latent_pil in offset_latents_pil] + [loss_fig, sigmas_fig]
        # grid_lbls = ['Input', 'Recovered'] + [name for name, _ in latents_pil] + [name for name, _ in offset_latents_pil] + ['Train Loss', 'Sigma']

        # grid = make_grid(grid_imgs, labels=grid_lbls, cols = 1, rows = len(grid_imgs), scale = 1)
        # grid.save(f'{output_path}/summary.png')
        loss_fig.save(f'{output_path}/loss.png')
        sigmas_fig.save(f'{output_path}/sigmas.png')
        


    states, gts_raw = None, None
    n_frames = cfg.n_frames
    val_gts, val_gts_raw = None, None

    if cfg.use_LRM:
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
        dataloader = DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=1)

    else:
        if cfg.load_exr:
            cfg.use_pre_rendered_video = False


        if cfg.use_pre_rendered_video:
            assert cfg.tonemapped.lower() != 'none'
            gts = unpack_video_to_numpy(f'{cfg.pre_rendered_dataset_path}/btf_sequence.mp4')
            n_frames = len(gts) # IMPORTANT: if refactored this, remember to return this.
            
            # gts = [np.asarray(Image.fromarray((gt*255.0).astype(np.uint8)).resize((cfg.resolution, cfg.resolution), Image.Resampling.LANCZOS))/255.0 for gt in gts]
        


        else:
            if cfg.load_exr:
                gts_raw = [mi.TensorXf(mi.Bitmap(f'{cfg.material_path}/exr/btf_frame__{i:04d}.exr')).numpy() for i in range(n_frames)]
                gts = [tonemapping(cfg.tonemapped, gt) for gt in gts_raw]
                
            else:
                gts = [(np.asarray(Image.open(f'{cfg.material_path}/png/btf_frame__{i:04d}.png'))/255.0) for i in range(n_frames)]
        gts = [torch.from_numpy(gt).to(device) for gt in gts]
        if gts_raw is not None:
            gts_raw = [torch.from_numpy(gt).to(device) for gt in gts_raw]
        
        if gts[0].shape[0] != cfg.resolution:
            gts = [torch.nn.functional.interpolate(gt.permute(2, 0 , 1).unsqueeze(0), size=(cfg.resolution, cfg.resolution), mode='bilinear').squeeze(0).permute(1, 2, 0).to(device) for gt in gts]
            if gts_raw is not None:
                gts_raw = [torch.nn.functional.interpolate(gt.permute(2, 0 , 1).unsqueeze(0), size=(cfg.resolution, cfg.resolution), mode='bilinear').squeeze(0).permute(1, 2, 0).to(device) for gt in gts_raw]


    if cfg.scene_state_path is None:
        states = [np.load(f'{cfg.material_path}/scene_state/btf_frame__{i:04d}.npy', allow_pickle=True).item() for i in range(n_frames)]
    else:
        states = [np.load(f'{cfg.scene_state_path}/{n_frames} frames/btf_frame__{i:04d}.npy', allow_pickle=True).item() for i in range(n_frames)]


    if cfg.optimize:
        # load scene state and ground truth
        
        if cfg.use_LRM:
            LRM.freeze_except(['dit'])
            torch_optim = torch.optim.AdamW(list(neural_brdf.parameters()) +  list(LRM.dit.parameters()) , lr = cfg.lr)

        else:
            torch_optim = torch.optim.Adam(neural_brdf.parameters(), lr = cfg.lr)

        loss_values, sigma_values = [], []

        pbar = tqdm(total=len(dataloader))
        half_life = cfg.n_iterations // 3






        for step, data in enumerate(dataloader):
            # step += 1 
            sigma = int(math.exp(-step/half_life)*4)*2 + 1
            if not cfg.use_LRM:
                with torch.no_grad():
                    neural_brdf.latents.data = gaussian_blur(neural_brdf.latents, kernel_size= sigma)
                    if neural_brdf.use_offset:
                        neural_brdf.offset_latents.data = gaussian_blur(neural_brdf.offset_latents, kernel_size= sigma+2) # bigger sigma for the offset
                sigma_values.append(sigma)
            else:
                gts = [torch.from_numpy(np.asarray(frame).astype(np.float32)/255.0).to(device) for frame in data['video']]
                # gts = torch.stack(gts, dim=0)



                gts = torch.stack([gts[i] for i in range(0,len(gts), 8)], dim=0)
                this_states = [states[i] for i in range(0,len(states), 8)]
                data['video'] = [data['video'][i] for i in range(0,len(data['video']), 8)]
                data['prompt'] = ''


                os.makedirs(f'{output_path}/dumps/step_{step}', exist_ok=True)


                for i in range(len(data['video'])):                    
                    to_pil(gts[i]).save(f'{output_path}/dumps/step_{step}/gt_{i}.png')
                    data['video'][i].save(f'{output_path}/dumps/step_{step}/video_{i}.png')


                # if neural_brdf.latents is not None:
                #     neural_brdf.latents = neural_brdf.latents.detach()
                #     if neural_brdf.use_offset:
                #         neural_brdf.offset_latents = neural_brdf.offset_latents.detach()

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    lrm_output_og = eval_lrm(LRM, data)
                
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()


                lrm_output = lrm_output_og
                latents, offset_latents = lrm_output.chunk(2, dim = 1)
                print('shape of latents and offset_latents', latents.shape, offset_latents.shape)
                neural_brdf.latents = latents.float()

                if neural_brdf.use_offset:                    
                    neural_brdf.offset_latents = offset_latents.float()

            mi.Bitmap(mi.TensorXf(latents[0, 0:3].permute(1, 2, 0).detach())).write(f'{output_path}/latents_{step}.exr')

            if (step) % 2000 == 0:
                os.makedirs(f'{output_path}/step_{step}', exist_ok=True)
                validation(cfg, neural_brdf, f'{output_path}/step_{step}', gts, gts_raw, val_gts, val_gts_raw, loss_values, sigma_values)


            # frame_ind = np.random.choice(range(n_frames))
            # gt = gts[frame_ind]
            grad_lrm_out = 0.0

            n_frames_this = len(this_states)
            n_frames_this = 1

            b = 1
            for j in range(math.ceil(n_frames_this/b)):
                interval = [j*b, min((j+1)*b, n_frames_this)]
                print('interval', interval)

                outputs = render_torch(
                        states=this_states, 
                        frame_ind=interval, 
                        geometry=geometry, 
                        neural_brdf=neural_brdf, 
                        resolution=cfg.resolution, 
                        device=device, 
                        seed=step,
                        convert_to_point_light=True
                    )


                



                img_raw = outputs['image']
                vis_mask = (outputs['visibility'] > 0.999999)[...,0].detach() # shape = H,w

                img = tonemapping(cfg.tonemapped, img_raw)



                # mi.Bitmap(mi.TensorXf(img)).write(f'{output_path}/img_{step}_{frame_ind}.exr')
                # mi.Bitmap(mi.TensorXf(gt)).write(f'{output_path}/gt_{step}_{frame_ind}.exr')


                loss, loss_pp = calc_loss(img, gts[interval[0]:interval[1]], cfg.loss_func, vis_mask, None)

                # loss = torch.mean((img - gt)**2)

                if not cfg.use_LRM and cfg.use_offset and cfg.offset_lambda is not None:
                    lambdaa = 1 / (step/cfg.offset_lambda + 1) 
                    loss+=lambdaa*(outputs['brdf_offset']).abs().mean()

                # loss.backward()
                # grad_lrm_out += torch.autograd.grad(loss, lrm_output, retain_graph=False)[0]
                loss.backward()

            # print(grad_lrm_out.shape, lrm_output.shape,grad_lrm_out.device, lrm_output.device)



            # lrm_output_og.backward(grad_lrm_out, retain_graph=False)

            # neural_brdf.latents = neural_brdf.latents.detach()
            # if neural_brdf.use_offset:
            #     neural_brdf.offset_latents = neural_brdf.offset_latents.detach()
            # lrm_output = lrm_output.detach()
            

            torch_optim.step()
            torch_optim.zero_grad()


            del grad_lrm_out, lrm_output_og, lrm_output, latents, offset_latents


            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            gc.collect()


            loss_values.append(loss.detach().item())
            pbar.set_postfix(loss=f"{loss_values[-1]:.4f}", sigma=sigma)
            pbar.update(1)

        os.makedirs(f'{output_path}/final', exist_ok=True)
        validation(cfg, neural_brdf, f'{output_path}/final', gts, gts_raw, val_gts, val_gts_raw, loss_values, sigma_values)
        np.save(f'{output_path}/loss_values.npy', loss_values)


    def final_validation(cfg, neural_brdf, output_path, gts, gts_raw, val_gts, val_gts_raw, loss_values, sigma_values):

        torch.set_grad_enabled(False)
        if cfg.training_video:
            print('Training video...')
            torch.cuda.empty_cache()
            training_renderings = []
            for i in range(n_frames):
                img = render_torch(states=states, frame_ind=i, geometry=geometry, neural_brdf=neural_brdf, resolution=cfg.resolution, device=device, seed=0, spp=4, convert_to_point_light=True)['image']
                dr.flush_malloc_cache()
                dr.flush_kernel_cache()
                training_renderings.append(img.cpu().detach().numpy())


            train_imgs_def = [np.array(mi.Bitmap(val).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)) for val in training_renderings]
            train_imgs_opt = [(np.array(mi.Bitmap(tonemapping(cfg.tonemapped, val).clip(min=0.0, max=1.0)))*255.0).astype(np.uint8) for val in training_renderings]
            gts = [(np.array(mi.Bitmap(mi.TensorXf(gt)))*255.0).astype(np.uint8) for gt in gts]


            # show_triplet_video(gts, train_imgs_def, train_imgs_opt, ['Input', 'Recovered default TM', f'Recovered optimiziation TM {cfg.tonemapped}'], fps=24, save_path=f'{output_path}/train_views.mp4', srgb_1=False, srgb_2=False, srgb_3=False)
            show_video_from_mitsuba_frames(train_imgs_def, fps=24, save_path=f'{output_path}/train_views_srgb.mp4')
            show_video_from_mitsuba_frames(train_imgs_opt, fps=24, save_path=f'{output_path}/train_views_optimTM.mp4')
            # gt_to_save=gts
            # if gts_raw is not None and cfg.tonemapped.lower() == 'none':
            #     gt_to_save=gts_raw
            show_video_from_mitsuba_frames(gts, fps=24, save_path=f'{output_path}/input_video.mp4')

        if cfg.validation_video:
            print('Validating...')
            torch.cuda.empty_cache()
            val_renderings = []
            n_frames = len(val_states)
            for i in range(n_frames):
                img = render_torch(states=val_states, frame_ind=i, geometry=geometry, neural_brdf=neural_brdf, resolution=cfg.resolution, device=device, seed=0, spp=4, convert_to_point_light=True)['image']
                dr.flush_malloc_cache()
                dr.flush_kernel_cache()
                val_renderings.append(img.cpu().detach().numpy())
            
            val_imgs_def = [np.array(mi.Bitmap(val).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)) for val in val_renderings]
            val_imgs_opt = [(np.array(mi.Bitmap(tonemapping(cfg.tonemapped, val).clip(min=0.0, max=1.0)))*255.0).astype(np.uint8) for val in val_renderings]
            show_video_from_mitsuba_frames(val_imgs_def, fps=24, save_path=f'{output_path}/val_views_srgb.mp4')
            show_video_from_mitsuba_frames(val_imgs_opt, fps=24, save_path=f'{output_path}/val_views_optimTM.mp4')


        if cfg.shader_ball_video:
            print('Rendering shader ball video...')
            import src.misc.render_matprev
            neural_brdf.network.network.uv_scale = 1
            src.render_matprev.render_matprev(neural_brdf, 8, save_path = f'{output_path}/fitted_matprev.mp4', resolution=cfg.resolution)
        
        if cfg.offset_video:
            # val_imgs_def = unpack_video_to_numpy(f'{output_path}/../../../val_views_srgb.mp4')
            # train_imgs_def = unpack_video_to_numpy(f'{output_path}/../../../train_views_srgb.mp4')        
            # val_imgs_def = [(val*255.0).astype(np.uint8) for val in val_imgs_def]
            # train_imgs_def = [(val*255.0).astype(np.uint8) for val in train_imgs_def]

            print('Rendering offset video...')
            

            neural_brdf.return_offset_as_separate_output = True

            def offset_video_render_and_blend(background_imgs, states, name):
                torch.cuda.empty_cache()
                offset_renderings = []
                for i in range(len(background_imgs)):
                    img = render_torch(states=states, frame_ind=i, geometry=geometry, neural_brdf=neural_brdf, resolution=cfg.resolution, device=device, seed=0, spp=4, convert_to_point_light=True)['brdf_offset']
                    if i == 0:
                        mi.util.write_bitmap(f'{output_path}/{name}_offset_render_{i}.exr', img)
                    dr.flush_malloc_cache()
                    dr.flush_kernel_cache()
                    offset_renderings.append(img.cpu().detach().numpy())
                print('rendered')

                def blend_offset_with_image(shift, img, channel):
                    a = Image.fromarray((colormap_diverging(shift[...,channel], -0.01, 0.01)*255.0).astype(np.uint8))
                    b = Image.fromarray((np.array(img)).astype(np.uint8))
                    return np.asarray(Image.blend(a, b, 0.2))
                print('blending')

                offset_shift_x = [blend_offset_with_image(val, img, 0) for val, img in zip(offset_renderings, background_imgs)]
                offset_shift_y = [blend_offset_with_image(val, img, 1) for val, img in zip(offset_renderings, background_imgs)]
                print('saving')
                
                show_video_from_mitsuba_frames(offset_shift_x, fps=24, save_path=f'{output_path}/{name}_offset_shift_x.mp4')
                show_video_from_mitsuba_frames(offset_shift_y, fps=24, save_path=f'{output_path}/{name}_offset_shift_y.mp4')
            
            offset_video_render_and_blend(train_imgs_def, states, 'train_views')
            offset_video_render_and_blend(val_imgs_def, val_states, 'val_views')






if __name__ == '__main__':
    main()





