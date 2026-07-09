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
from src.neumip_mitsuba import NeuralBSDF, MitsubaTorchWrapperMLP, load_latents, NeuralBSDFOffsetOnly
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
        # params[-1]['area_light_sphere.to_world'] = (mi.Transform4f().translate([-2,0,3])).matrix.numpy().tolist()
        # params[-1]['area_light_sphere.emitter.radiance.value']  = np.asarray([30,30,30])
        params[-1]['sensor.to_world'] = mi.Transform4f().look_at(
                                        origin=camera_pos[i].tolist(),
                                        target=[0, 0, 0],
                                        up=[0, 1- t[i].item(), t[i].item()],
                                    ).matrix.numpy().tolist()
        params[-1]['sensor.x_fov'] = 30

    return params




@hydra.main(version_base=None, config_path="config", config_name="inverse_render")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve = True))
    print(f"Working directory : {os.getcwd()}")
    output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output directory  : @@@{output_path}@@@")

    renderer = StreamingBTFRenderer(resolution=cfg.resolution, grid_size=cfg.mesh_grid_size, only_center_mesh_instance=cfg.only_center_mesh_instance, arealight=cfg.use_area_light)
    

    val_states = validation_cameras(lightgeight = renderer.lightgeight, area_light_radius = renderer.area_light_radius, camera_height = renderer.camera_height, base_light_intensity = renderer.base_light_intensity, camera_radius = renderer.camera_radius, use_area_light=cfg.use_area_light)




    dr.set_flag(dr.JitFlag.LoopRecord, False)
    dr.set_flag(dr.JitFlag.VCallRecord, False)
    dr.set_flag(dr.JitFlag.VCallOptimize, False)




    mi.register_bsdf("neural", lambda props: NeuralBSDF(props))


    neural_brdf = mi.load_dict({'type': 'neural'})
    neural_brdf.network = MitsubaTorchWrapperMLP({"use_offset": cfg.use_offset, "resolution": cfg.resolution, "latent_channels": cfg.latent_channels, "network_width": cfg.network_width, "network_depth": cfg.network_depth})


    if cfg.use_checkpoint:
        print('Loading checkpoint...')
        train_cfg = OmegaConf.load(Path(cfg.checkpoint) / "../../.hydra/config.yaml")
        cfg.network_width = train_cfg.network_width
        cfg.network_depth = train_cfg.network_depth
        cfg.latent_channels = train_cfg.latent_channels
        cfg.use_offset = train_cfg.use_offset
        neural_brdf.network = MitsubaTorchWrapperMLP({"use_offset": cfg.use_offset, "resolution": cfg.resolution, "latent_channels": cfg.latent_channels, "network_width": cfg.network_width, "network_depth": cfg.network_depth})

        

        neural_brdf.network.network.load_state_dict(torch.load(cfg.checkpoint))
        print('Checkpoint loaded')

    renderer.scene_dict['my_shape_group']['material_plane']['bsdf'] = neural_brdf
    scene = mi.load_dict(renderer.scene_dict)
    params = mi.traverse(scene)






    def validation(cfg,scene, params, integrator, neural_brdf, output_path, gts, gts_raw, loss_values, sigma_values):
        frame_ind = 0
        load_scene_state(states[frame_ind], params, transfer_area_light_to_point_light=not cfg.use_area_light)      
        gt = gts[frame_ind]
        save_exrs = 'final' in output_path

        with torch.no_grad():
            with dr.suspend_grad():
                img_raw = mi.render(scene = scene, params = params, integrator = integrator, spp= 4, seed = 0)
                img = tonemapping(cfg.tonemapped, img_raw)

        if save_exrs:
            mi.util.write_bitmap(f'{output_path}/fitted_linear.exr', mi.Bitmap(img_raw))
            if gts_raw is not None:
                mi.util.write_bitmap(f'{output_path}/gt_linear.exr', mi.Bitmap(gts_raw[frame_ind]))
        # to_pil = lambda x: Image.fromarray((x.numpy()*255.0).astype(np.uint8))
        from src.utils import to_pil
        img_pil = to_pil(img)
        gt_pil = to_pil(gt)
        img_pil.save(f'{output_path}/fitted.png')
        gt_pil.save(f'{output_path}/gt.png')

        latent_div = range(math.ceil(cfg.latent_channels/3))

        get_latents = lambda x,i: x.detach().cpu().numpy()[0,3*i:3*i+3].transpose(1,2,0)
        offset_latents, offset_latents_pil = [], []
        latents = [(f'latent_{3*i}-{min(3*i+3, cfg.latent_channels)}', get_latents(neural_brdf.network.network.latents, i)) for i in latent_div]
        if neural_brdf.network.network.use_offset:
            offset_latents = [(f'offset_latent_{3*i}-{min(3*i+3, cfg.latent_channels)}', get_latents(neural_brdf.network.network.offset_latents, i)) for i in latent_div]

        latents_pil = [(name, to_pil(np.abs(_lat)*10)) for name, _lat in latents]
        if neural_brdf.network.network.use_offset:
            offset_latents_pil = [(name, to_pil(np.abs(_lat)*10)) for name, _lat in offset_latents]


        if save_exrs:
            [mi.util.write_bitmap(f'{output_path}/{name}.exr', mi.Bitmap(_lat)) for name, _lat in latents]   
            if neural_brdf.network.network.use_offset:
                [mi.util.write_bitmap(f'{output_path}/{name}.exr', mi.Bitmap(_lat)) for name, _lat in offset_latents]


        # [mi.util.write_bitmap(f'{output_path}/latent_{3*i}-{3*i+3}.exr', mi.Bitmap(neural_brdf.network.network.latents.detach().cpu().numpy()[0,3*i:3*i+3].transpose(1,2,0))) for i in latent_div]
        # mi.util.write_bitmap(f'{output_path}/fitted_latent_6.exr', mi.Bitmap(neural_brdf.network.network.latents.detach().cpu().numpy()[0,6:].transpose(1,2,0)))
            # mi.util.write_bitmap(f'{output_path}/fitted_offset_latent_6.exr', mi.Bitmap(neural_brdf.network.network.offset_latents.detach().cpu().numpy()[6].transpose(1,2,0)))


        torch.save(neural_brdf.network.network.state_dict(), f'{output_path}/model.pth')

        loss_fig = plot_loss(np.asarray(loss_values), list(range(len(loss_values))), resolution=(800,800), ylabel='Loss', xlabel = 'Epochs', legend = 'Loss', color = 'blue', font_size = None, linewidth = 2)
        sigmas_fig = plot_loss(np.asarray(sigma_values), list(range(len(sigma_values))), resolution=(800,800), ylabel='Sigma', xlabel = 'Steps', legend = 'Sigma', color = 'blue', font_size = None, linewidth = 2, y_log_scale = False)
        
        grid_imgs = [gt_pil, img_pil] + [latent_pil for _, latent_pil in latents_pil] + [offset_latent_pil for _, offset_latent_pil in offset_latents_pil] + [loss_fig, sigmas_fig]
        grid_lbls = ['Input', 'Recovered'] + [name for name, _ in latents_pil] + [name for name, _ in offset_latents_pil] + ['Train Loss', 'Sigma']

        # grid = make_grid(grid_imgs, labels=grid_lbls, cols = 1, rows = len(grid_imgs), scale = 1)
        # grid.save(f'{output_path}/summary.png')
        loss_fig.save(f'{output_path}/loss.png')
        sigmas_fig.save(f'{output_path}/sigmas.png')
        
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

    def compute_loss(img, gt, loss_func):
        if loss_func == 'L2':
            return (img - dr.detach(gt))**2
        elif loss_func == 'L2rel':
            return ((img - dr.detach(gt))**2) / (dr.detach(gt ** 2) + 0.01)
        elif loss_func == 'L2log':
            return (dr.log(1+dr.maximum(img, 0.0)) - dr.log(1+dr.detach(gt)))**2
        else:
            raise ValueError(f"Invalid loss function: {loss_func}")

    integrator = mi.load_dict({'type': 'direct', 'bsdf_samples': 1, 'emitter_samples': 1})

    states, gts_raw = None, None
    n_frames = cfg.n_frames


    if cfg.load_exr:
        cfg.use_pre_rendered_video = False


    if cfg.use_pre_rendered_video:
        assert cfg.tonemapped.lower() != 'none'
        gts = unpack_video_to_numpy(f'{cfg.pre_rendered_dataset_path}/btf_sequence.mp4')
        n_frames = len(gts) # IMPORTANT: if refactored this, remember to return this.
        
        gts = [np.asarray(Image.fromarray((gt*255.0).astype(np.uint8)).resize((cfg.resolution, cfg.resolution), Image.Resampling.LANCZOS))/255.0 for gt in gts]
    
        gts = [mi.TensorXf(gt) for gt in gts]



    else:
        if cfg.load_exr:
            gts_raw = [mi.TensorXf(mi.Bitmap(f'{cfg.material_path}/exr/btf_frame__{i:04d}.exr')) for i in range(n_frames)]
            gts = [tonemapping(cfg.tonemapped, gt) for gt in gts_raw]
            
        else:
            gts = [mi.TensorXf(np.asarray(Image.open(f'{cfg.material_path}/png/btf_frame__{i:04d}.png'))/255.0) for i in range(n_frames)]

    if cfg.scene_state_path is None:
        states = [np.load(f'{cfg.material_path}/scene_state/btf_frame__{i:04d}.npy', allow_pickle=True).item() for i in range(n_frames)]
    else:
        states = [np.load(f'{cfg.scene_state_path}/{n_frames} frames/btf_frame__{i:04d}.npy', allow_pickle=True).item() for i in range(n_frames)]


    class CamRayIntegrator(mi.SamplingIntegrator):
        def sample(self: mi.SamplingIntegrator, scene: mi.Scene, sampler, ray: mi.RayDifferential3f, medium: mi.Medium = None, active: bool = True):
            si:mi.SurfaceInteraction3f = scene.ray_intersect(ray, mi.RayFlags.All, True, active)
            valid_ray = active & si.is_valid()
            result = dr.select(valid_ray, mi.Vector3f(1), mi.Vector3f(0))
            return result, active, []
    
    vis_int = CamRayIntegrator(mi.Properties())
    vis_maks = []
    for state in states:
        load_scene_state(state, params, transfer_area_light_to_point_light=not cfg.use_area_light)
        vis_mask = mi.render(scene, integrator=vis_int, spp=16)
        vis_mask = dr.select(vis_mask > 0.999999, mi.Float(1), mi.Float(0))
        vis_maks.append(vis_mask)



    if cfg.optimize:
        # load scene state and ground truth


        



        dr.enable_grad(params['my_shape_group.material_plane.bsdf.network'])
        torch_optim = torch.optim.Adam(neural_brdf.network.network.parameters(), lr = cfg.lr)





        loss_values, sigma_values = [], []

        pbar = tqdm(total=cfg.n_iterations+1)

        # step = 0
        half_life = cfg.n_iterations // 3
        # frame_list = list(range(n_frames))
        # np.random.shuffle(frame_list)
        # epoch_loss = 0


        def calc_loss(image, image_ref, loss_func, masks, weights):
            if loss_func.lower() == 'l2':
                loss_img =  (dr.square(image - image_ref))
            elif loss_func.lower() == 'l2rel':
                loss_img = (dr.square(image - image_ref) / dr.detach(dr.square(image_ref) + 0.01))
            elif loss_func.lower() == 'l2log':
                loss_img = dr.square(dr.log(1+image) - dr.log(1+dr.detach(image_ref)))
            else:
                raise Exception('loss not recognized')

            if weights is not None:
                loss_img = dr.detach(weights) * loss_img

            if masks is None:
                return dr.mean(loss_img), loss_img
            else:
                # torch_mask =torch.cat([masks.torch()>0.5]*3, dim = -1) 
                # a = (torch_mask.view(-1).nonzero().view(-1).cpu().numpy())
                # b = mi.UInt64(a)
                # assert len(b)%3 == 0
                # new_loss = dr.gather(mi.Float, loss_img.array, b)    
                # loss = dr.mean(new_loss) * 3
                count = dr.sum(masks)
                loss_img = loss_img * masks
                loss = dr.sum(loss_img)/count

            return loss, loss_img


        for step in range(cfg.n_iterations+1):
            # step += 1 
            sigma = int(dr.exp(-step/half_life)*4)*2 + 1
            with torch.no_grad():
                neural_brdf.network.network.latents.data = gaussian_blur(neural_brdf.network.network.latents, kernel_size= sigma)
                if neural_brdf.network.network.use_offset:
                    neural_brdf.network.network.offset_latents.data = gaussian_blur(neural_brdf.network.network.offset_latents, kernel_size= sigma+2) # bigger sigma for the offset
            sigma_values.append(sigma)



            if (step) % 2000 == 0:
                os.makedirs(f'{output_path}/step_{step}', exist_ok=True)
                validation(cfg, scene, params, integrator, neural_brdf, f'{output_path}/step_{step}', gts, gts_raw, loss_values, sigma_values)




            frame_ind = np.random.choice(range(n_frames))
            load_scene_state(states[frame_ind], params, transfer_area_light_to_point_light=not cfg.use_area_light)      
            gt = gts[frame_ind]
            

            img_raw = mi.render(scene = scene, params = params, integrator = integrator, spp= 4, seed = step)

            img = tonemapping(cfg.tonemapped, img_raw)
            
            # loss_pp = compute_loss(img, gt, cfg.loss_func)
            loss, loss_pp = calc_loss(img, gt, cfg.loss_func, vis_maks[frame_ind], None)




            # loss = dr.mean(loss_pp)
            dr.backward(loss)

            # epoch_loss+=loss.numpy().item()

            torch_optim.step()
            torch_optim.zero_grad()

            loss_values.append(loss.numpy().item())
            pbar.set_postfix(loss=f"{loss_values[-1]:.4f}", sigma=sigma, frame=frame_ind)
            pbar.update(1)  

        os.makedirs(f'{output_path}/final', exist_ok=True)
        validation(cfg, scene, params, integrator, neural_brdf, f'{output_path}/final', gts, gts_raw, loss_values, sigma_values)
        np.save(f'{output_path}/loss_values.npy', loss_values)



    torch.set_grad_enabled(False)
    dr.disable_grad(params)
    if cfg.training_video:
        print('Training video...')
        torch.cuda.empty_cache()
        training_renderings = []
        for i in range(n_frames):
            load_scene_state(states[i], params, transfer_area_light_to_point_light=not cfg.use_area_light)
            img = mi.render(scene, integrator = integrator, params = params, spp=4)
            dr.flush_malloc_cache()
            dr.flush_kernel_cache()
            training_renderings.append(img.numpy())


        train_imgs_def = [np.array(mi.Bitmap(val).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)) for val in training_renderings]
        train_imgs_opt = [(np.array(mi.Bitmap(tonemapping(cfg.tonemapped, mi.TensorXf(val))))*255.0).astype(np.uint8) for val in training_renderings]
        gts = [(np.array(mi.Bitmap(gt))*255.0).astype(np.uint8) for gt in gts]


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
            load_scene_state(val_states[i], params, transfer_area_light_to_point_light=not cfg.use_area_light)
            img = mi.render(scene, integrator = integrator, params = params, spp=4)
            dr.flush_malloc_cache()
            dr.flush_kernel_cache()
            val_renderings.append(img.numpy())
        
        val_imgs_def = [np.array(mi.Bitmap(val).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)) for val in val_renderings]
        val_imgs_opt = [(np.array(mi.Bitmap(tonemapping(cfg.tonemapped, mi.TensorXf(val))))*255.0).astype(np.uint8) for val in val_renderings]
        show_video_from_mitsuba_frames(val_imgs_def, fps=24, save_path=f'{output_path}/val_views_srgb.mp4')
        show_video_from_mitsuba_frames(val_imgs_opt, fps=24, save_path=f'{output_path}/val_views_optimTM.mp4')


    if cfg.shader_ball_video:
        print('Rendering shader ball video...')
        import src.render_matprev
        neural_brdf.network.network.uv_scale = 1
        src.render_matprev.render_matprev(neural_brdf, 8, save_path = f'{output_path}/fitted_matprev.mp4', resolution=cfg.resolution)
    
    if cfg.offset_video:
        # val_imgs_def = unpack_video_to_numpy(f'{output_path}/../../../val_views_srgb.mp4')
        # train_imgs_def = unpack_video_to_numpy(f'{output_path}/../../../train_views_srgb.mp4')        
        # val_imgs_def = [(val*255.0).astype(np.uint8) for val in val_imgs_def]
        # train_imgs_def = [(val*255.0).astype(np.uint8) for val in train_imgs_def]

        print('Rendering offset video...')
        class OffsetIntegrator(mi.SamplingIntegrator):
            def sample(self: mi.SamplingIntegrator, scene: mi.Scene, sampler, ray: mi.RayDifferential3f, medium: mi.Medium = None, active: bool = True):
                si:mi.SurfaceInteraction3f = scene.ray_intersect(ray, mi.RayFlags.All, True, active)
                valid_ray = active & si.is_valid()
                # si.wi is passed instead of wo, which is fine because we will factor out the cosine, and also it does not have an effect on the offset value
                offset_value = si.bsdf(ray).eval(mi.BSDFContext(), si, si.wi, active)

                result = dr.select(valid_ray, mi.Vector3f(offset_value), mi.Vector3f(0))
                return result, active, []
        
        mi.register_bsdf("neural_offset_only", lambda props: NeuralBSDFOffsetOnly(props))
        neural_brdf_offset_only = mi.load_dict({'type': 'neural_offset_only'})
        neural_brdf_offset_only.network = neural_brdf.network

        renderer.scene_dict['my_shape_group']['material_plane']['bsdf'] = neural_brdf_offset_only
        scene_offset_only = mi.load_dict(renderer.scene_dict)
        params_offset_only = mi.traverse(scene_offset_only)
        neural_brdf_offset_only.network.network.return_offset = True

        def offset_video_render_and_blend(background_imgs, states, name):
            torch.cuda.empty_cache()
            offset_renderings = []
            for i in range(len(background_imgs)):
                load_scene_state(states[i], params_offset_only, transfer_area_light_to_point_light=not cfg.use_area_light)
                img = mi.render(scene_offset_only, integrator = OffsetIntegrator(mi.Properties()), params = params_offset_only, spp=4)
                if i == 0:
                    mi.util.write_bitmap(f'{output_path}/{name}_offset_render_{i}.exr', img)
                dr.flush_malloc_cache()
                dr.flush_kernel_cache()
                offset_renderings.append(img.numpy())
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





