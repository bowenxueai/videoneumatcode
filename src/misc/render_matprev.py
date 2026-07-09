import torch
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import tempfile, base64, os
import numpy as np
import imageio
from IPython.display import HTML
import mitsuba as mi
import cv2
import drjit as dr
import shutil

def get_scene_dict(material = None, resolution = 1024):
    scene_dict = {
        "type": "scene",
        "integrator": {
            "type": "path",
            "max_depth": -1,
        },

        "sensor": {
            "type": "perspective",
            "id": "camera",
            "fov_axis": "smaller",
            "focus_distance": 6.0,
            "fov": 28.8415,
            "to_world": mi.ScalarTransform4f().look_at(
                origin=[3.69558, -3.46243, 3.25463],
                target=[3.04072, -2.85176, 2.80939],
                up=[-0.317366, 0.312466, 0.895346]
            ),
            "sampler": {
                "type": "independent",
                "sample_count": 64
            },
            "film": {
                "type": "hdrfilm",
                "id": "film",
                "width": resolution,
                "height": resolution,
                "pixel_format": "rgb",
                "rfilter": {"type": "gaussian"}
            },
        },

        # "emitter": {
        #     "type": "envmap",
        #     "id": "emitter-envmap",
        #     "filename": "/home/shadadan/NeuGen/data/scenes/matpreview/envmap.exr",
        #     "to_world": (
                
        #         mi.ScalarTransform4f(
        #             [[-0.224951, -0.000001, -0.974370, 0.000000],
        #             [-0.974370, 0.000000, 0.224951, 0.000000],
        #             [0.000000, 1.000000, -0.000001, 8.870000],
        #             [0.000000, 0.000000, 0.000000, 1.000000]]
        #         ).rotate([0, 1, 0], -180)
        #     ),
        #     "scale": 1.0
        # },

        'area_light': {
            'type': 'disk',
            # 'center': [2,2,2],
            # 'radius': 0.5,
            'emitter': {
                'type': 'area',
                'radiance': {
                    'type': 'rgb',
                    'value': 50.0,
                }
            }
        },

        "bsdf-diffuse": {
            "type": "diffuse",
            "id": "bsdf-diffuse",
            "reflectance": {"type": "rgb", "value": [0.18, 0.18, 0.18]},
        },

        "texture-checkerboard": {
            "type": "checkerboard",
            "id": "texture-checkerboard",
            "color0": {"type": "rgb", "value": 0.4},
            "color1": {"type": "rgb", "value": 0.2},
            "to_uv": mi.ScalarTransform4f().scale([8.0, 8.0, 1.0]),
        },

        "bsdf-plane": {
            "type": "diffuse",
            "id": "bsdf-plane",
            "reflectance": {"type": "ref", "id": "texture-checkerboard"},
        },

        "bsdf-matpreview": {
            "type": "plastic",
            "id": "bsdf-matpreview",
            "diffuse_reflectance": {"type": "rgb", "value": [0.240, 0.771, 0.361]},
            "int_ior": 1.9,
        },

        # Uncomment this block if you want to include the plane:
        # "shape-plane": {
        #     "type": "serialized",
        #     "id": "shape-plane",
        #     "filename": "matpreview.serialized",
        #     "shape_index": 0,
        #     "to_world": (
        #         mi.ScalarTransform4f.rotate([0, 0, 1], -4.3) @
        #         mi.ScalarTransform4f([
        #             3.38818, -4.06354, 0, -1.74958,
        #             4.06354, 3.38818, 0, 1.43683,
        #             0, 0, 5.29076, -0.0120714,
        #             0, 0, 0, 1
        #         ])
        #     ),
        #     "bsdf": {"type": "ref", "id": "bsdf-plane"},
        # },

        "shape-matpreview-interior": {
            "type": "serialized",
            "id": "shape-matpreview-interior",
            "filename": "/home/shadadan/NeuGen/data/scenes/matpreview/matpreview.serialized",
            "shape_index": 1,
            "to_world": mi.ScalarTransform4f([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0.0252155],
                [0, 0, 0, 1]
            ]),
            "bsdf": {"type": "ref", "id": "bsdf-diffuse"},
        },

        "shape-matpreview-exterior": {
            "type": "serialized",
            "id": "shape-matpreview-exterior",
            "filename": "/home/shadadan/NeuGen/data/scenes/matpreview/matpreview.serialized",
            "shape_index": 2,
            "to_world": (
                mi.ScalarTransform4f([
                    [0.614046, 0.614047, 0, -1.78814e-07],
                    [-0.614047, 0.614046, 0, 2.08616e-07],
                    [0, 0, 0.868393, 1.02569],
                    [0, 0, 0, 1]
                ]) @ mi.ScalarTransform4f().translate([0, 0, 0.01])
            ),
            "bsdf": {"type": "ref", "id": "bsdf-matpreview"},
        },
    }


    if material is not None:
        scene_dict["shape-matpreview-exterior"]['bsdf'] = material

    return scene_dict



import math, numpy as np
from random import seed

def generate_camera_locations(n_views, from_to, radius, elevation, object_bbox, seed = 0, fov = 45):

    '''
        This function generates camera location around the object
        Currently many things are hardcoded for the bunny and teapot scenes, need to generalize in the future

        input:
            n_views: int, number of the views
            from_to: list[int, int], for modes mv360 and cylinder, if not None, means only render [from, to] views if the view was (hypothetically) 128 
            seed: int, the seed to generate random camera locations
            mode: a str how to choose the cameras
            object_bbox: a dict
            radius: how far is the camera from the object
            elevation: for 360 mode and cylinder mode: the z of the camera

        output:
            origins: a numpy array hodling camera locations

    '''

    # TODO: save camera locations as well
    # TODO: add a default camera intrinsics
    
    generator = np.random.default_rng(seed)
 
 

    # a 360 rotation around trhe object
    t = ((np.arange(n_views)) / (n_views))
    tshifted = ((np.arange(n_views) + int(0.1*n_views)) / (n_views))

    # a = range(0,128, 8)
    # t = np.asarray([t[_a] for _a in a])

    if from_to:
        t = (np.linspace(from_to[0], from_to[1], n_views) + 5) / 128
        # b = (np.linspace(from_to[1], from_to[0], n_views//2) + 5) / 128
        # t = np.concatenate([a,  b], axis = 0)

    dolly_amp = 0.3,      # e.g. 0.15 for ±15% radius modulation
    dolly_freq = 1,     # cycles over the whole path
    # FOV zoom (change perspective)
    do_fov_zoom: bool = False,
    fov_amp = 3,        # e.g. 3.0 degrees
    fov_freq = 1.0,       # cycles over the whole path
    fov0 = 30
    radius_scale = 1.0 - np.abs(dolly_amp * np.sin(t * np.pi * 2 * dolly_freq))
    fov = fov0 + (fov_amp * np.sin(t * np.pi * 2 * fov_freq)) if do_fov_zoom else [fov0]*n_views

    path_x = radius*np.cos(t * np.pi * 2) * radius_scale
    path_y = radius*np.sin(t * np.pi * 2) * radius_scale
    path_x_shifted = radius*np.cos(tshifted * np.pi * 2) * radius_scale
    path_y_shifted = radius*np.sin(tshifted * np.pi * 2) * radius_scale
    path_z = np.ones_like(t)* elevation       # just choose some altitude
    path = np.stack([path_x, path_y, path_z], axis = -1)
    path_shifted = np.stack([path_x_shifted, path_y_shifted, path_z + 2.5], axis = -1)

    # path = torch.nn.functional.normalize(torch.from_numpy(np.stack([path_x, path_y, path_z], axis = -1)), dim = -1).numpy()
    origins = (path + object_bbox["center"])
    origins_shifted = (path_shifted + object_bbox["center"])    


    up = np.zeros_like(origins)
    # up[:, 1] = 1
    up = np.asarray([0, 0, 0.868393]) + up
    lookat = np.zeros_like(origins) + object_bbox["center"]
    # fov = fov * np.ones(up.shape[0])

    cameras = {
        "origins": origins,
        "origins_shifted": origins_shifted,
        "fov": fov,
        "lookat": lookat,
        "up": up,
        "transforms": [mi.Transform4f().look_at(origin=o, target=l, up=u) for o, l, u in zip(origins, lookat, up)],
        "transforms_shifted": [mi.Transform4f().look_at(origin=o, target=l, up=u) for o, l, u in zip(origins_shifted, lookat, up)],

    }

    return cameras



def render_matprev(material = None, spp = 32, save_path = None,  resolution = 1024):

    scene_dict = get_scene_dict(material, resolution)

    scene = mi.load_dict(scene_dict)



    def get_object_bounding_box(shape):
        bounds = shape.bbox()
        return _get_bounds_dict(bounds)


    def _get_bounds_dict(bounds):
        bbox = {
            "center": bounds.center().numpy(),
            "max_bound": bounds.max.numpy(),
            "min_bound": bounds.min.numpy(),
        }
        return bbox


    bbox = get_object_bounding_box(scene.shapes()[-1])

    cameras = generate_camera_locations(
        n_views=240,
        from_to=None,
        radius=5.0,
        elevation=1,
        object_bbox=bbox
    )

    direct_integrator = mi.load_dict({'type': 'direct', 'bsdf_samples': 1, 'emitter_samples': 1, 'hide_emitters': True})

    params = mi.traverse(scene)
    imgs = []
    seed = 0
    for T, T_shifted, fov in zip(cameras["transforms"], cameras["transforms_shifted"], cameras["fov"]):
        seed+=1
        params["camera.to_world"] = T
        params["camera.x_fov"] = fov
        params["area_light.to_world"] = dr.copy(T_shifted).scale(0.5)
        params.update()
        with torch.no_grad():
            with dr.suspend_grad():
                img = mi.render(scene, integrator=direct_integrator, spp=spp ,seed = seed)
                imgs.append(img)
                torch.cuda.empty_cache()
                dr.flush_malloc_cache()
                torch.cuda.empty_cache()
                dr.flush_malloc_cache()




    def _to_uint8_rgb(frame: mi.TensorXf) -> np.ndarray:
        """Convert Mitsuba TensorXf → uint8 RGB numpy array."""
        bmp = mi.Bitmap(frame)
        bmp_u8 = bmp.convert(
            pixel_format=mi.Bitmap.PixelFormat.RGB,
            component_format=mi.Struct.Type.UInt8,
            srgb_gamma=True
        )
        return np.array(bmp_u8, copy=False)

    def show_video_from_mitsuba_frames(frames, fps=24, bitrate=None, resize=None, spp = 32, save_path = None):
        """
        Render a list of Mitsuba mi.TensorXf frames as an inline MP4 in Jupyter.
        """
        # Create a temp file (deleted after embedding)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            temp_path = tmpfile.name

        # Write the video to disk via FFmpeg
        writer = imageio.get_writer(
            temp_path,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            bitrate=bitrate or "8M"
        )

        for f in frames:
            rgb = _to_uint8_rgb(f)
            if resize is not None:
                rgb = cv2.resize(rgb, resize[::-1])
            writer.append_data(rgb)
        writer.close()
        if save_path is not None:
            shutil.copy(temp_path, save_path)

        # Read file into memory and embed it as base64
        with open(temp_path, "rb") as f:
            video_bytes = f.read()
        os.remove(temp_path)

        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        return HTML(f"""
        <video controls autoplay loop style="max-width: 100%;">
        <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
        """)



    return show_video_from_mitsuba_frames(imgs, fps=24, save_path = save_path)