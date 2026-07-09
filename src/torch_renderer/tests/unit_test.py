# %%
import numpy as np



import sys
sys.path.append('/home/shadadan/NeuGen')
sys.path.append('/home/shadadan/NeuGen/src')

import imageio.v3 as iio


from render import *
import mitsuba as mi
import drjit as dr
mi.set_variant('cuda_ad_rgb')
from typing import Tuple, List, Any, Optional
import drjit as dr
import gc
import inspect

spp = 1
jitter = False


output = '/home/shadadan/NeuGen/outputs/unit_test'
os.makedirs(output, exist_ok=True)


def get_path(i):
    path = f'{output}/{inspect.currentframe().f_back.f_code.co_name}/frame_{i:04d}/'
    os.makedirs(path, exist_ok=True)
    return path



from torch_renderer import (
    Plane,
    Texture,
    create_camera_from_mitsuba,
    DiffuseBRDF,
    render,
)

states = [np.load(f'/home/shadadan/NeuGen/data/matsynth_scene_state/{81} frames/btf_frame__{i:04d}.npy', allow_pickle=True).item() for i in range(81)]
texture = "/home/shadadan/NeuGen/notebooks/table.png"
width = 512
height = 512



def render_torch(spp, jitter, light_position, light_intensity, to_world, fov, height, width, grads=False, grads_mean=False, brdf=None):
    # Render
    apply_filter = True
    filter_stddev = 0.5
    filter_radius = 2.0


    # Torch renderer
    device = "cuda"
    device = torch.device(device)

    plane_width = 4.0
    plane_height = 4.0
    plane = Plane.create(width=plane_width, height=plane_height, device=device)


    sensor = {
        "fov": fov,  # allow overriding via fov
        "resolution": (height, width),
        "to_world": to_world,
    }
    camera = create_camera_from_mitsuba(sensor, device=device)

    # BRDF
    if not brdf:
        brdf = DiffuseBRDF(reflectance=texture, device=device, srgb=True)
        brdf.albedo.texture.requires_grad = grads

    light_intensity = torch.tensor(light_intensity, dtype=torch.float32, device=device)
    if light_intensity.ndim == 0:
        light_intensity = light_intensity.unsqueeze(0)
    
    light_intensity = light_intensity.repeat_interleave(3, dim=-1) if (light_intensity.shape[-1] ==1) else light_intensity
    seed = 0
    outputs = render(
            camera=camera,
            plane=plane,
            brdf=brdf,
            image_size=(height, width),
            light_position=torch.tensor(light_position.tolist(), dtype=torch.float32, device=device),
            light_intensity=light_intensity,
            apply_filter=apply_filter,
            filter_stddev=filter_stddev,
            filter_radius=filter_radius,
            spp=spp,
            jitter=jitter,
            seed=seed,
    )

    uv_torch = outputs["uv"]
    outputs["uv"] = torch.cat([uv_torch, torch.zeros_like(uv_torch)[...,0:1]], dim=-1)

    if grads:
        function = torch.mean if grads_mean else torch.sum
        function((outputs['image']-10000).square()).backward()
        torch_grad = brdf.albedo.texture.grad
        outputs['grad'] = torch_grad

    return outputs






renderer = StreamingBTFRenderer(resolution=512, grid_size=4, only_center_mesh_instance=True, arealight=False)

def setup_mitsuba_renderer(to_world, light_intensity, light_position, fov, spp, jitter):
    class CenterPixelSamplerIntegrator(mi.python.ad.integrators.common.ADIntegrator):
        # this integrator inherits from the mitsuba built-in ADIntegrator with one difference, 
        # it disables the random offset for antialising and only samples from the center of the pixel
        def sample_rays(
            self,
            scene: mi.Scene,
            sensor: mi.Sensor,
            sampler: mi.Sampler,
        ) -> Tuple[mi.RayDifferential3f, mi.Spectrum, mi.Vector2f, mi.Float]:
            """
            Sample a 2D grid of primary rays for a given sensor

            Returns a tuple containing

            - the set of sampled rays
            - a ray weight (usually 1 if the sensor's response function is sampled
            perfectly)
            - the continuous 2D image-space positions associated with each ray
            """
            film = sensor.film()
            film_size = film.crop_size()
            rfilter = film.rfilter()
            border_size = rfilter.border_size()

            if film.sample_border():
                film_size += 2 * border_size

            spp = sampler.sample_count()
            assert jitter or spp == 1 #NOTE: We assume 1 spp for now

            # Compute discrete sample position
            idx = dr.arange(mi.UInt32, dr.prod(film_size) * spp)

            # Try to avoid a division by an unknown constant if we can help it
            log_spp = dr.log2i(spp)
            if 1 << log_spp == spp:
                idx >>= dr.opaque(mi.UInt32, log_spp)
            else:
                idx //= dr.opaque(mi.UInt32, spp)

            # Compute the position on the image plane
            pos = mi.Vector2i()
            pos.y = idx // film_size[0]
            pos.x = dr.fma(mi.UInt32(mi.Int32(-film_size[0])), pos.y, idx)

            if film.sample_border():
                pos -= border_size

            pos += mi.Vector2i(film.crop_offset())

            # Cast to floating point and add random offset
            pos_f = mi.Vector2f(pos) + 0.5                    #NOTE: other than 1spp, the only difference to mitsuba built-in code is here, instead of random offset, we sample the center of the pixel
            if jitter:
                pos_f += sampler.next_2d() - 0.5
                
            # Re-scale the position to [0, 1]^2
            scale = dr.rcp(mi.ScalarVector2f(film.crop_size()))
            offset = -mi.ScalarVector2f(film.crop_offset()) * scale
            pos_adjusted = dr.fma(pos_f, scale, offset)

            aperture_sample = mi.Vector2f(0.0)
            if sensor.needs_aperture_sample():
                aperture_sample = sampler.next_2d()

            time = sensor.shutter_open()
            if sensor.shutter_open_time() > 0:
                time += sampler.next_1d() * sensor.shutter_open_time()

            wavelength_sample = 0
            if mi.is_spectral:
                wavelength_sample = sampler.next_1d()

            with dr.resume_grad():
                ray, weight = sensor.sample_ray_differential(
                    time=time,
                    sample1=wavelength_sample,
                    sample2=pos_adjusted,
                    sample3=aperture_sample
                )

            # With box filter, ignore random offset to prevent numerical instabilities
            splatting_pos = mi.Vector2f(pos) if rfilter.is_box_filter() else pos_f

            return ray, weight, splatting_pos

    scene_dict = renderer.scene_dict.copy()
    scene_dict['my_shape_group']['material_plane']['bsdf'] = mi.load_dict({'type': 'diffuse', 'reflectance': 
    #{'type': 'rgb', 'value': [0.8, 0.8, 0.8]}})
    {'type': 'bitmap', 'filename': texture,
                            'wrap_mode': 'mirror',
                            'filter_type': 'bilinear'
    }})

    # scene_dict['sensor']['to_world'] =  mi.ScalarTransform4f().look_at(origin=eye,target=at,up=up)
    # scene_dict['sensor']['to_world'] = mi.Transform4f(np.asarray(to_world))
    scene_dict['light']['intensity'] = {'type': 'rgb', 'value': light_intensity}
    scene_dict['light']['position'] = light_position
    scene = mi.load_dict(scene_dict)

    params = mi.traverse(scene)

    params['sensor.to_world'] = mi.Transform4f(np.asarray(to_world))
    params['sensor.x_fov'] = fov
    params.update()



    class VisibilityIntegrator(CenterPixelSamplerIntegrator):
        def sample(self,
                mode: dr.ADMode,
                scene: mi.Scene,
                sampler: mi.Sampler,
                ray: mi.Ray3f,
                depth: mi.UInt32,
                δL ,
                δaovs ,
                state_in,
                active: mi.Bool):

            si:mi.SurfaceInteraction3f = scene.ray_intersect(ray, mi.RayFlags.All, True, active)
            valid_ray = active & si.is_valid()
            result = dr.select(valid_ray, mi.Vector3f(1), mi.Vector3f(0))
            return result, active, [], None


    class UVIntegrator(CenterPixelSamplerIntegrator):
        def sample(self,
                mode: dr.ADMode,
                scene: mi.Scene,
                sampler: mi.Sampler,
                ray: mi.Ray3f,
                depth: mi.UInt32,
                δL ,
                δaovs ,
                state_in,
                active: mi.Bool):

            si:mi.SurfaceInteraction3f = scene.ray_intersect(ray, mi.RayFlags.All, True, active)
            valid_ray = active & si.is_valid()
            uv = np.concatenate([si.uv.numpy(), np.zeros_like(si.uv.numpy())[0:1]], axis=0)
            result = dr.select(valid_ray, mi.Vector3f(uv), mi.Vector3f(0))
            return result, active, [], None

    class AlbedoIntegrator(CenterPixelSamplerIntegrator):
        def sample(self,
                mode: dr.ADMode,
                scene: mi.Scene,
                sampler: mi.Sampler,
                ray: mi.Ray3f,
                depth: mi.UInt32,
                δL ,
                δaovs ,
                state_in,
                active: mi.Bool):

            si:mi.SurfaceInteraction3f = scene.ray_intersect(ray, mi.RayFlags.All, True, active)
            valid_ray = active & si.is_valid()

            result = si.bsdf().eval_diffuse_reflectance(si, active)

            result = dr.select(valid_ray, result, mi.Vector3f(0))
            return result, active, [], None



    class CamRayIntegrator(CenterPixelSamplerIntegrator):
        def sample(self,
                mode: dr.ADMode,
                scene: mi.Scene,
                sampler: mi.Sampler,
                ray: mi.Ray3f,
                depth: mi.UInt32,
                δL ,
                δaovs ,
                state_in,
                active: mi.Bool):

            si:mi.SurfaceInteraction3f = scene.ray_intersect(ray, mi.RayFlags.All, True, active)
            valid_ray = active & si.is_valid()
            result = dr.select(valid_ray, ray.d, mi.Vector3f(0))
            return result, active, [], None

    class LightRayIntegrator(CenterPixelSamplerIntegrator):
        def sample(self,
                mode: dr.ADMode,
                scene: mi.Scene,
                sampler: mi.Sampler,
                ray: mi.Ray3f,
                depth: mi.UInt32,
                δL ,
                δaovs ,
                state_in,
                active: mi.Bool):

            si:mi.SurfaceInteraction3f = scene.ray_intersect(ray, mi.RayFlags.All, True, active)
            result = mi.Spectrum(0.0)

            # NEE
            ctx = mi.BSDFContext()
            bsdf = si.bsdf(ray)
            flags = bsdf.flags()
            active = active & si.is_valid()

            sample_emitter = active & mi.has_flag(flags, mi.BSDFFlags.Smooth)

            active_e = sample_emitter
            (ds, emitter_val) = scene.sample_emitter_direction(si, sampler.next_2d(active_e), True, active_e)
            active_e = active & (ds.pdf != 0.0)

            result = dr.select(active_e, ds.d, mi.Vector3f(0))

            return result, active, [], None

    mi_direct_integraotr = mi.load_dict({'type': 'direct', 'bsdf_samples': 0, 'emitter_samples': 1})


    class direct(CenterPixelSamplerIntegrator):
        def sample(self,
                mode: dr.ADMode,
                scene: mi.Scene,
                sampler: mi.Sampler,
                ray: mi.Ray3f,
                depth: mi.UInt32,
                δL ,
                δaovs ,
                state_in,
                active: mi.Bool):

            result , active , _ = mi_direct_integraotr.sample(scene, sampler, ray,  None ,active)

            return result, active, [], None




    visibility_integrator = VisibilityIntegrator(mi.Properties())
    direct_integrator = direct(mi.Properties())
    uv_integrator = UVIntegrator(mi.Properties())
    albedo_integrator = AlbedoIntegrator(mi.Properties())
    cam_ray_integrator = CamRayIntegrator(mi.Properties())
    light_ray_integrator = LightRayIntegrator(mi.Properties())

    integrators = {
        'visibility': visibility_integrator,
        'image': direct_integrator,
        'uv': uv_integrator,
        'albedo': albedo_integrator,
        'cam_dir': cam_ray_integrator,
        'light_dir': light_ray_integrator,
        "mi_direct": mi_direct_integraotr,
    }

    return scene, params, integrators

def render_mitsuba(scene, params, integrators, spp,grads=False):
    if grads:
        dr.enable_grad(params['my_shape_group.material_plane.bsdf.reflectance.data'])

    image_mi = (mi.render(scene, params, integrator = integrators["image"] if not grads else integrators["mi_direct"], spp = spp))


    visibility_mi = (mi.render(scene, integrator = integrators["visibility"], spp = spp))
    visibility_mi = visibility_mi.numpy()[:,:,0:1]
    uv_mi = (mi.render(scene, integrator = integrators["uv"], spp = spp))
    albedo_mi = (mi.render(scene, integrator = integrators["albedo"], spp = spp))
    cam_ray_mi = (mi.render(scene, integrator = integrators["cam_dir"], spp = spp))
    light_dir_mi = (mi.render(scene, integrator = integrators["light_dir"], spp = spp))

    outputs = {
        'image': image_mi,
        'visibility': visibility_mi,
        'uv': uv_mi,
        'albedo': albedo_mi,
        'cam_dir': cam_ray_mi,
        'light_dir': light_dir_mi,
    }

    if grads:
        loss = dr.mean(dr.sqr(image_mi - 10000))
        dr.backward(loss)
        mi_grad = dr.grad(params['my_shape_group.material_plane.bsdf.reflectance.data'])
        outputs['grad'] = mi_grad        


    return outputs


def fetch_states():
    to_world = torch.tensor([states[i]['sensor.to_world'] for i in range(len(states))])[..., 0]
    light_position = torch.tensor([np.asarray(states[i]['area_light_sphere.to_world'])[:3, -1, 0] for i in range(len(states))])
    fov = torch.tensor([states[i]['sensor.x_fov'] for i in range(len(states))])
    light_intensity = torch.tensor([np.asarray(states[i]['area_light_sphere.emitter.radiance.value'])[...,0] for i in range(len(states))]) / 10
    return to_world, light_position, fov, light_intensity



def test_forward():
    print("============= Test Forward 1 SPP no jitter =============")
    to_world, light_position, fov, light_intensity = fetch_states()
    for i in range(len(states)):
        print(f"=============================== frame {i+1} of {len(states)} ================")

        path = get_path(i)

        light_intensity = 40.1685

        spp = 1
        jitter = False
        outputs = render_torch(spp, jitter, light_position[i], light_intensity, to_world[i], fov[i], height, width, grads=False)

        scene, params, integrators = setup_mitsuba_renderer(to_world=np.asarray(to_world[i]), light_intensity=(light_intensity), light_position=np.asarray(light_position[i]), fov=np.asarray(fov[i]), spp=spp, jitter=jitter)
        outputs_mi = render_mitsuba(scene, params, integrators, spp, grads=False)

        comparison(i, outputs_mi, outputs, save_exrs=False, tolerance=0.001, path = path)


def test_forward_batched():
    print("============= Test Forward 1 SPP no jitter batched =============")


    to_world, light_position, fov, light_intensity = fetch_states()
    spp = 1
    jitter = False
    outputs = render_torch(spp, jitter, light_position, light_intensity, to_world, fov, height, width, grads=False)

    for i in range(len(states)):
        print(f"=============================== frame {i+1} of {len(states)} ================")

        path = get_path(i)


        scene, params, integrators = setup_mitsuba_renderer(to_world=np.asarray(to_world[i]), light_intensity=np.asarray(light_intensity[i]), light_position=np.asarray(light_position[i]), fov=fov[i], spp=spp, jitter=jitter)
        outputs_mi = render_mitsuba(scene, params, integrators, spp, grads=False)


        outputs_torch_this = {key: value[i] for key, value in outputs.items()}

        comparison(i, outputs_mi, outputs_torch_this, save_exrs=False, tolerance=0.0011, path = path)



def test_forward_multi_spp_batched():
    print("============= Test Forward Multi SPP Batched =============")


    to_world, light_position, fov, light_intensity = fetch_states()

    spp = 128
    jitter = True

    all_batches = []
    b = 4
    c = 0
    for i in range(len(states)//b):
        outputs_batch = render_torch(spp, jitter, light_position[i*b:(i+1)*b], light_intensity[i*b:(i+1)*b], to_world[i*b:(i+1)*b], fov[i*b:(i+1)*b], height, width, grads=False)
        all_batches.append(outputs_batch)
        c+=1
    
    outputs = {key: torch.cat([outputs_batch[key] for outputs_batch in all_batches], dim=0) for key in all_batches[0]}
    
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    gc.collect()

    for i in range(c):
        print(f"=============================== frame {i+1} of {len(states)} ================")

        path = get_path(i)


        scene, params, integrators = setup_mitsuba_renderer(to_world=np.asarray(to_world[i]), light_intensity=np.asarray(light_intensity[i]), light_position=np.asarray(light_position[i]), fov=fov[i], spp=spp, jitter=jitter)
        outputs_mi = render_mitsuba(scene, params, integrators, spp, grads=False)


        outputs_torch_this = {key: value[i] for key, value in outputs.items()}

        comparison(i, outputs_mi, outputs_torch_this, save_exrs=False, tolerance=0.01, path = path)



def test_forward_multi_spp():
    print("============= Test Forward Multi SPP + jitter =============")
    to_world, light_position, fov, light_intensity = fetch_states()

    for i in range(len(states)):
        print(f"=============================== frame {i+1} of {len(states)} ================")

        path = get_path(i)

        light_intensity = 40.1685

        spp = 128
        jitter = True
        outputs = render_torch(spp=spp, jitter=jitter, light_position=light_position[i], light_intensity=light_intensity, to_world=to_world[i], fov=fov[i], height=height, width=width, grads=False)
        scene, params, integrators = setup_mitsuba_renderer(to_world=np.asarray(to_world[i]), light_intensity=(light_intensity), light_position=np.asarray(light_position[i]), fov=np.asarray(fov[i]), spp=spp, jitter=jitter)
        outputs_mi = render_mitsuba(scene, params, integrators, spp, grads=False)

        comparison(i, outputs_mi, outputs, save_exrs=False, tolerance=0.01, path = path)



def test_backward_torch_batched():
    print("============= Test Backward torch batched vs torch unbatched  =============")


    to_world, light_position, fov, light_intensity = fetch_states()

    spp = 4
    jitter = False

    b = 81
    for i in range(len(states)//b):
        print(f"=============================== batch {i} of {len(states)//b} from {i*b} to {(i+1)*b} ================")
        outputs_batch = render_torch(spp, jitter, light_position[i*b:(i+1)*b], light_intensity[i*b:(i+1)*b], to_world[i*b:(i+1)*b], fov[i*b:(i+1)*b], height, width, grads=True)
        grads_batched = outputs_batch['grad'].detach().clone()

        grads_accum = 0

        for j in range(b):
            view_id = i*b + j
            print(f"=============================== frame {view_id} of {len(states)} ================")

            path = get_path(view_id)

            grads_accum += render_torch(spp, jitter, light_position[view_id], light_intensity[view_id], to_world[view_id], fov[view_id], height, width, grads=True)['grad'].detach().clone()

        assert torch.allclose(grads_accum, grads_batched), f"Gradients are not equal {grads_accum.shape} {grads_batched.shape} {torch.abs(grads_accum - grads_batched).mean()}"


def test_backward_torch_batched_grad_accumulation():
    print("============= Test Backward torch batched vs torch unbatched grad_accumulation =============")


    to_world, light_position, fov, light_intensity = fetch_states()

    spp = 4
    jitter = False

    brdf = DiffuseBRDF(reflectance=texture, device='cuda', srgb=True)
    brdf.albedo.texture.requires_grad = True
    b = 8
    for i in range(len(states)//b):
        print(f"=============================== batch {i} of {len(states)//b} from {i*b} to {(i+1)*b} ================")
        outputs_batch = render_torch(spp, jitter, light_position[i*b:(i+1)*b], light_intensity[i*b:(i+1)*b], to_world[i*b:(i+1)*b], fov[i*b:(i+1)*b], height, width, grads=False, brdf=brdf)
        (outputs_batch['image'] - 10000).square().sum().backward()
    grads_batched_accum = brdf.albedo.texture.grad.detach().clone()

    grads_non_batched_accum = 0

    for i in range(len(states)//b):
        for j in range(b):
            view_id = i*b + j
            print(f"=============================== frame {view_id} of {len(states)} ================")
            path = get_path(view_id)
            grads_non_batched_accum += render_torch(spp, jitter, light_position[view_id], light_intensity[view_id], to_world[view_id], fov[view_id], height, width, grads=True)['grad'].detach().clone()

    assert torch.allclose(grads_batched_accum, grads_non_batched_accum)



def test_forward_torch_batched():
    print("============= Test forward torch batched vs. torch unbatched =============")


    to_world, light_position, fov, light_intensity = fetch_states()

    spp = 4
    jitter = False

    b = 81
    for i in range(len(states)//b):
        print(f"=============================== batch {i} of {len(states)//b} from {i*b} to {(i+1)*b} ================")
        outputs = render_torch(spp, jitter, light_position[i*b:(i+1)*b], light_intensity[i*b:(i+1)*b], to_world[i*b:(i+1)*b], fov[i*b:(i+1)*b], height, width, grads=False)


        for j in range(b):
            view_id = i*b + j
            print(f"=============================== frame {view_id} of {len(states)} ================")

            path = get_path(view_id)

            outputs_this = render_torch(spp, jitter, light_position[view_id], light_intensity[view_id], to_world[view_id], fov[view_id], height, width, grads=False)
            outputs_batched_this = {key: value[j].detach().cpu() for key, value in outputs.items()}

            comparison(view_id, outputs_batched_this, outputs_this, save_exrs=False, tolerance=0.00000, path = path)







def test_backward_multi_spp():
    print("============= Test Forward Multi SPP + jitter =============")
    to_world, light_position, fov, light_intensity = fetch_states()

    for i in range(len(states)):
        print(f"=============================== frame {i+1} of {len(states)} ================")

        path = get_path(i)


        light_intensity = 40.1685

        spp = 128
        jitter = True
        outputs = render_torch(spp=spp, jitter=jitter, light_position=light_position[i], light_intensity=light_intensity, to_world=to_world[i], fov=fov[i], height=height, width=width, grads=True, grads_mean=True)

        scene, params, integrators = setup_mitsuba_renderer(to_world=np.asarray(to_world[i]), light_intensity=(light_intensity), light_position=np.asarray(light_position[i]), fov=np.asarray(fov[i]), spp=spp, jitter=jitter)
        outputs_mi = render_mitsuba(scene, params, integrators, spp, grads=True)



        comparison(i, outputs_mi, outputs, save_exrs=False, tolerance=0.01, path = path)








def comparison(view_id, mi_outputs, torch_outputs, tolerance=0.001, save_exrs=False, path = None):

    keys = ['image', 'uv', 'visibility', 'albedo', 'cam_dir', 'light_dir', 'grad']

    # maps = {'image': [image_mi, image_torch], 'uv': [uv_mi, uv_torch], 'visibility': [visibility_mi, visibility_torch], 'albedo': [albedo_mi, albedo_reflectance_torch], 'cam_dir': [cam_ray_mi, cam_dir_torch], 'light_dir': [light_dir_mi, light_dir_torch], 'grad': [mi_grad, torch_grad]}



    for key in keys:

        if key not in mi_outputs or key not in torch_outputs:
            continue


        a = np.array(mi_outputs[key])
        b = np.array(torch_outputs[key].detach().cpu())
        error = float(np.abs(a - b).mean())

        if save_exrs:
            if key == 'grad':
                a = np.abs(a)*100
                b = np.abs(b)*100
            mi.Bitmap(a).write(f'{path}/{key}_mi.exr')
            mi.Bitmap(b).write(f'{path}/{key}_torch.exr')


        assert error <= tolerance, f"Error for {key} view {view_id} is {error}, which is greater than tolerance {tolerance}"



if __name__ == '__main__':
    test_forward()
    test_forward_batched()
    test_forward_torch_batched()
    test_forward_multi_spp()
    test_forward_multi_spp_batched()
    test_backward_torch_batched()
    test_backward_multi_spp()
    test_backward_torch_batched_grad_accumulation()

# # %%
