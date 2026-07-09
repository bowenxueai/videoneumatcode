import mitsuba as mi
import numpy as np
from pathlib import Path
import json
import cv2
import imageio
import os
import drjit as dr
import torch
import torchvision.transforms.functional as TF
from datasets import load_dataset
from torch.utils.data import DataLoader
import gc
import multiprocessing as mp
from functools import partial
from scipy.ndimage import gaussian_filter
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import math
import csv
import argparse
from utils import load_scene_state

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    from btf_sensor import BTFSensor
    print("BTF sensor loaded successfully")
except ImportError as e:
    print(f"Error: Could not import BTF sensor")
    raise

mi.set_variant('cuda_ad_rgb')

if not hasattr(mi, '_clamped_path_registered'):
    class ClampedPath(mi.SamplingIntegrator):
        def __init__(self, props):
            super().__init__(props)
            self.clamp = props.get("radiance_clamp", 10.0)
            self.base = mi.load_dict({
                "type": "path",
                "max_depth": props.get("max_depth", 6),
                "hide_emitters": props.get("hide_emitters", True)
            })
        
        def sample(self, scene, sampler, ray, medium=None, active=True):
            spec, mask, *aovs = self.base.sample(scene, sampler, ray, medium, active)
            spec_clamped = dr.clip(spec, 0.0, 10)
            return (spec_clamped, mask, *aovs)
        
        def aov_names(self):
            return self.base.aov_names() if hasattr(self.base, 'aov_names') else []
        
        def traverse(self, callback):
            super().traverse(callback)
            callback.put('radiance_clamp', self.clamp, mi.ParamFlags.NonDifferentiable)
    
    mi.register_integrator("clamped_path", lambda p: ClampedPath(p))
    mi._clamped_path_registered = True
    print("Clamped path integrator registered successfully")


def validation_cameras(total_n_frames=320, area_light_radius=0.25, lightgeight=2, 
                       camera_height=4, base_light_intensity=None, 
                       camera_radius=4.0, use_area_light=True):
    """生成验证用的相机和光源轨迹 - 只存储原始位置数据"""
    if base_light_intensity is None:
        base_light_intensity = [204.5762567336908, 204.5762567336908, 204.5762567336908]
    
    params = []
    n_frames = total_n_frames // 2

    t0 = np.linspace(-20, 20, n_frames // 2)
    t1 = np.linspace(20, -3, n_frames - n_frames // 2)
    ty = np.concatenate([t0, 0 * t1])
    tx = np.concatenate([0 * t0, t1])

    lightpos = np.stack([
        tx,
        ty,
        tx * 0 + (lightgeight * 2 + 3)
    ], axis=-1)

    actual_distance = np.sqrt(np.sum(lightpos ** 2, axis=-1))
    base_distance = lightgeight * 2 + 2

    intensity_correction = (actual_distance / base_distance) ** 2
    corrected_intensity = np.array(base_light_intensity)[None] * intensity_correction[:, None]
    last_light_energy = None
    last_light_pos = None

    for i in range(n_frames):
        params.append({})
        last_light_energy = corrected_intensity[i].tolist()
        last_light_pos = lightpos[i].tolist()
        
        params[-1]['light_intensity'] = last_light_energy
        params[-1]['light_pos'] = last_light_pos
        params[-1]['camera_pos'] = [0, 0, camera_height + 2]
        params[-1]['camera_up'] = [0, 1, 0]
        params[-1]['sensor.x_fov'] = 30

    n_frames_phase2 = total_n_frames - total_n_frames // 2

    t = np.linspace(0.5, 1.5, n_frames_phase2)
    camera_pos = np.stack([
        camera_radius * 0.4 * np.cos(t * 2 * np.pi),
        camera_radius * 0.4 * np.sin(t * 2 * np.pi),
        (np.cos((t) * np.pi) * camera_height + camera_height + 2)
    ], axis=-1)

    camera_pos[:, 0] += camera_radius * 0.4

    for i in range(n_frames_phase2):
        params.append({})
        params[-1]['light_intensity'] = last_light_energy
        params[-1]['light_pos'] = last_light_pos
        params[-1]['camera_pos'] = camera_pos[i].tolist()
        params[-1]['camera_up'] = [0, float(1 - t[i]), float(t[i])]
        params[-1]['sensor.x_fov'] = 30

    return params



def dense_cameras(total_n_frames=320, area_light_radius=0.25, lightgeight=2, 
                       camera_height=4, base_light_intensity=None, 
                       camera_radius=4.0, use_area_light=True):
    """生成验证用的相机和光源轨迹 - 只存储原始位置数据"""
    if base_light_intensity is None:
        base_light_intensity = [204.5762567336908, 204.5762567336908, 204.5762567336908]
    
    params = []
    n_root = 4
    n_frames = n_root**4

    x, y = np.linspace(0.2,0.8,n_root), np.linspace(0.1,0.9,n_root)
    xyzk = np.stack(np.meshgrid(x, x,y,y, indexing='xy'), axis = -1).reshape(-1,4).astype(np.float32)
    xyz = mi.warp.square_to_uniform_hemisphere(mi.Vector2f(xyzk[:,:2].transpose(1,0))).numpy()
    abc = mi.warp.square_to_uniform_hemisphere(mi.Vector2f(xyzk[:,2:4].transpose(1,0))).numpy()
    xyzabc = np.concatenate([xyz, abc], axis = 0)
    hemisphere_radius = np.sqrt(4**2+4**2)/2
    hemisphere_radius = lightgeight * 2+ 2
    lightpos = xyzabc[:3, :].transpose(1,0)*hemisphere_radius
    camera_pos = xyzabc[3:6, :].transpose(1,0) * (camera_height + 2)



    actual_distance = np.sqrt(np.sum(lightpos ** 2, axis=-1))
    base_distance = hemisphere_radius

    intensity_correction = (actual_distance / base_distance) ** 2
    corrected_intensity = np.array(base_light_intensity)[None] * intensity_correction[:, None]
    
    # last_light_energy = None
    # last_light_pos = None

    for i in range(n_frames):
        params.append({})
        # last_light_energy = corrected_intensity[i].tolist()
        # last_light_pos = lightpos[i].tolist()
        
        params[-1]['light_intensity'] = corrected_intensity[i].tolist()
        params[-1]['light_pos'] = lightpos[i].tolist()
        params[-1]['camera_pos'] = camera_pos[i].tolist()
        params[-1]['camera_up'] = [0, 1, 0]
        params[-1]['sensor.x_fov'] = 30


    return params





def sample_states_uniform(states, target_n_frames):
    """从 states 中均匀采样 target_n_frames 帧"""
    total = len(states)
    if target_n_frames >= total:
        return states
    indices = np.linspace(0, total - 1, target_n_frames, dtype=int)
    return [states[i] for i in indices]


class StreamingBTFRenderer:

    def __init__(self, base_mesh_path=None, resolution=1024, base_size=4.0, gpu_id=0):
        self.cached_height_map = None 
        self.resolution = resolution
        self.base_size = base_size
        self.gpu_id = gpu_id
        self.calibration_output_dir = "./btf_dataset_output7"
        self.arealight = True
        self.camera_transform = mi.ScalarTransform4f().look_at(origin=[0, 0, 5.5], target=[0, 0, 0], up=[0, 1, 0])

        self.default_disp_scale = 1
        mi.set_variant(f'cuda_ad_rgb')
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
        
        print(f"Initializing BTF renderer on GPU {gpu_id} with resolution: {resolution}x{resolution}")
        print(f"Estimated texture memory per material: {resolution*resolution*3*4*5/(1024**3):.2f} GB")
        self.basemeshsize = self.resolution
        if base_mesh_path is None:
            self.base_mesh_path = self._create_base_mesh(grid_size=self.resolution)
        else:
            self.base_mesh_path = base_mesh_path

        self.scene_template = None
        self.params_template = None
        self.original_vertices = None
        self.camera = False        
        self.camera_radius = 4.0
        self.light_radius = 2.0
        self.camera_height = 4
        self.lightgeight = 2
        
        self.gaussian_sigma = 0.5
        
        self.area_light_radius = 0.25
        self.area_light_radiance = [204.5762567336908, 204.5762567336908, 204.5762567336908]

        self._create_scene_template()
        calibrated_intensity = self.calibrate_light_intensity()
        self.base_light_intensity = calibrated_intensity
        print("calibrated_intensity", calibrated_intensity)
        self.radiance_clamp = 15 / 256
        print("radiance_clamp", self.radiance_clamp)
        
        self.use_optix_denoiser = True  

        self.res = mi.ScalarVector2u(int(self.resolution), int(self.resolution))        
        self.denoiser = mi.OptixDenoiser(self.res, albedo=True, normals=True, temporal=False)

        # 用于记录位移信息
        self.displacement_stats = {}

    def calibrate_light_intensity(self):
        if self.scene_template is None:
            self._create_scene_template()

        print("\nAll parameters:")
        for key in self.params_template.keys():
            print(f"  {key}")
        
        print("\nLight parameters:")
        for key in self.params_template.keys():
            if 'light' in key.lower():
                print(f"  {key}")

        test_material = {
            'basecolor': torch.full((3, self.resolution, self.resolution), 0.2, dtype=torch.float32),
            'roughness': torch.ones(1, self.resolution, self.resolution, dtype=torch.float32),  
            'metallic': torch.zeros(3, self.resolution, self.resolution, dtype=torch.float32)
        }
        
        if self.calibration_output_dir:
            output_path = Path(self.calibration_output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            basecolor_np = test_material['basecolor'].permute(1, 2, 0).cpu().numpy()
            basecolor_path = output_path / "calibration_basecolor.exr"
            mi.util.write_bitmap(str(basecolor_path), basecolor_np)
            print(f"  Saved calibration basecolor to: {basecolor_path}")
            
        self.update_material(test_material)
        
        if self.arealight:
            transform = mi.Transform4f()
            transform = transform.translate(mi.Point3f(float(0), float(0), float(self.lightgeight * 2 + 2)))
            transform = transform.scale(self.area_light_radius)
            self.params_template['area_light_sphere.to_world'] = transform
        else:
            self.params_template['light.position'] = mi.Vector3f(
                float(0), float(0), float(self.lightgeight * 2 + 2)
            )
        
        test_intensity = [5.23984641071269, 5.23984641071269, 5.23984641071269]
        if self.arealight:
            self.params_template['area_light_sphere.emitter.radiance.value'] = mi.Vector3f(*test_intensity)
        else:
            self.params_template['light.intensity.value'] = mi.Vector3f(*test_intensity)

        self.params_template.update()
        
        image = mi.render(self.scene_template)
        beauty = image[..., :3]
        
        if self.calibration_output_dir:
            render_path = output_path / "calibration_render.exr"
            mi.util.write_bitmap(str(render_path), image)
            print(f"  Saved calibration render to: {render_path}")
        
        center_size = 10
        center_start = self.resolution // 2 - center_size // 2
        center_end = center_start + center_size
        center_region = beauty[center_start:center_end, center_start:center_end]
        measured_value = np.mean(center_region)
        
        correction_factor = 0.2 / measured_value
        calibrated_intensity = [float(val * correction_factor) for val in test_intensity]
        
        print(f"Correction factor calculation:")
        print(f"  measured_value: {measured_value:.3f}")
        print(f"  correction_factor: {correction_factor:.3f}")
        print(f"  calibrated_intensity: {calibrated_intensity}")
        
        return calibrated_intensity

    def _gaussian_blur_wrap_torch(self, height_np: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0:
            return height_np.astype(np.float32, copy=True)

        ksize = int(math.ceil(sigma * 6)) | 1
        pad = ksize // 2

        device = torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu')

        x = torch.arange(ksize, device=device, dtype=torch.float32) - (ksize - 1) / 2
        g = torch.exp(-0.5 * (x / sigma) ** 2)
        g = (g / g.sum()).to(torch.float32)

        img = torch.as_tensor(height_np, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img = F.pad(img, (pad, pad, pad, pad), mode='circular')
        img = F.conv2d(img, g.view(1, 1, ksize, 1))
        img = F.conv2d(img, g.view(1, 1, 1, ksize))

        return img[0, 0].contiguous().detach().cpu().numpy()

    def _linear_to_srgb_gpu(self, x):
        a = 0.055
        x_pos = dr.maximum(x, 0.0) 
        y = dr.select(
            x_pos <= 0.0031308,
            12.92 * x_pos,
            (1 + a) * (x_pos ** (1 / 2.4)) - a  
        )
        return dr.clip(y, 0.0, 1.0)
       
    def _read_height_factor(self, meta_dict=None, json_dir=None):
        if isinstance(meta_dict, dict) and 'height_factor' in meta_dict:
            try:
                return float(meta_dict['height_factor'])
            except Exception:
                pass

        if json_dir is not None:
            meta_path = Path(json_dir) / 'metadata.json'
            if meta_path.exists():
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    if 'height_factor' in meta:
                        return float(meta['height_factor'])
                except Exception as e:
                    print(f"[warn] metadata.json read error:{e}")

        return 1.0    
        
    def _create_base_mesh(self, grid_size=1024):
        print(f"Creating base mesh with {grid_size}x{grid_size} vertices...")
        
        x = np.linspace(-self.base_size / 2, self.base_size / 2, grid_size)
        y = np.linspace(-self.base_size / 2, self.base_size / 2, grid_size)
        xx, yy = np.meshgrid(x, y)
        print(f"Base size: {self.base_size}")
        print(f"X range: [{x.min():.2f}, {x.max():.2f}], width: {x.max() - x.min():.2f}")
        print(f"Y range: [{y.min():.2f}, {y.max():.2f}], height: {y.max() - y.min():.2f}")
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False)
        
        tex_w = tex_h = grid_size  
        eps_u = 0.5 / tex_w
        eps_v = 0.5 / tex_h

        for i in range(grid_size):
            for j in range(grid_size):
                temp_file.write(f"v {xx[i,j]} {yy[i,j]} 0\n")

        for i in range(grid_size):
            for j in range(grid_size):
                u = j / (grid_size - 1)
                v = 1 - i / (grid_size - 1)
                temp_file.write(f"vt {u} {v}\n")
        
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                v0 = i * grid_size + j + 1
                v1 = v0 + 1
                v2 = v0 + grid_size
                v3 = v2 + 1
                
                temp_file.write(f"f {v0}/{v0} {v1}/{v1} {v2}/{v2}\n")
                temp_file.write(f"f {v1}/{v1} {v3}/{v3} {v2}/{v2}\n")
        
        temp_file.close()
        print(f"Base mesh created: {temp_file.name}")
        return temp_file.name
        
    def _create_scene_template(self):
        """Create scene template with mixed mirror and rotation pattern"""
        move = 1 - 8 / 1024
        dummy_texture = mi.TensorXf(np.ones((self.resolution, self.resolution, 3), dtype=np.float32))
        
        material_plane_dict = {
            'type': 'obj',
            'filename': self.base_mesh_path,
            'bsdf': {
                'type': 'twosided', 
                'material': {
                    'type': 'principled',
                    'base_color': {
                        'type': 'bitmap',
                        'data': dummy_texture,
                        'wrap_mode': 'mirror',
                        'filter_type': 'bilinear'
                    },
                    'roughness': {
                        'type': 'bitmap',
                        'data': dummy_texture[...,0:1],
                        'raw': True,
                        'wrap_mode': 'mirror',
                        'filter_type': 'bilinear'
                    },
                    'metallic': {
                        'type': 'bitmap',
                        'data': dummy_texture[...,0:1],
                        'raw': True,
                        'wrap_mode': 'mirror',
                        'filter_type': 'bilinear'
                    }
                }
            }
        }
        
        scene_dict = {
            'type': 'scene',
            'sampler': {
                'type': 'ldsampler',
            },
            'my_shape_group': {
                'type': 'shapegroup',
                'material_plane': material_plane_dict
            },
            'area_light_sphere': {
                'type': 'sphere',
                'emitter': {
                    'type': 'area',
                    'radiance': {
                        'type': 'rgb',
                        'value': self.area_light_radiance
                    }
                }
            },
            'sensor': {
                'type': 'perspective',
                'fov': 37,
                'to_world': self.camera_transform,
                'film': {
                    'type': 'hdrfilm',
                    'width': self.resolution,
                    'height': self.resolution,
                    'pixel_format': 'rgb',
                    'component_format': 'float32',
                }
            },
            "integrator": {
                "type": "aov",
                "aovs": "albedo:albedo, normals:sh_normal",
                "integrator": {
                    "type": "clamped_path",
                    "max_depth": 4,
                    "radiance_clamp": 2,
                    "hide_emitters": True
                }
            }
        }

        mixed_configs = [
            (0, 0, 1, 1, 0),       
            (1, 0, 1, 1, 0),
            (-1, 0, 1, 1, 0),
            (0, move, 1, 1, 0),
            (0, -move, 1, 1, 0),
            (1, move, 1, 1, 0),
            (-1, move, 1, 1, 0),
            (1, -move, 1, 1, 0),
            (-1, -move, 1, 1, 0),
        ]

        tile_spacing = self.base_size 

        for idx, (i, j, scale_x, scale_y, angle) in enumerate(mixed_configs):
            transform = mi.ScalarTransform4f()
            transform = transform.scale([scale_x, scale_y, 1])
            if angle != 0:
                transform = transform.rotate([0, 0, 1], angle)

            offset_x = i * tile_spacing
            offset_y = j * tile_spacing

            ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            ox, oy = offset_x, offset_y
            ox, oy = (ca * ox + sa * oy, -sa * ox + ca * oy)
            if scale_x != 0: ox /= scale_x
            if scale_y != 0: oy /= scale_y

            transform = mi.ScalarTransform4f()
            transform = transform.scale([scale_x, scale_y, 1])
            if angle != 0:
                transform = transform.rotate([0, 0, 1], angle)
            transform = transform.translate([ox, oy, 0])   
            
            y_top = +self.base_size * 0.5
            y_bottom = -self.base_size * 0.5
            p_top = mi.ScalarPoint3f(0.0, y_top, 0.0)
            p_bottom = mi.ScalarPoint3f(0.0, y_bottom, 0.0)
            wp_top = transform @ p_top
            wp_bottom = transform @ p_bottom
            print(f"{idx}: i={i},j={j} -> world y(top)={wp_top.y:.6f}, y(bottom)={wp_bottom.y:.6f}")
        
            scene_dict[f'instance_mixed_{idx}'] = {
                'type': 'instance',
                'to_world': transform,
                'shapegroup': {
                    'type': 'ref',
                    'id': 'my_shape_group'
                }
            }

        self.scene_template = mi.load_dict(scene_dict)
        self.params_template = mi.traverse(self.scene_template)
        self.original_vertices = dr.copy(self.params_template['my_shape_group.material_plane.vertex_positions'])
        
        print(f"Scene template created with 3x3 mixed mirror/rotation pattern")
        print(f"  - Tile spacing: {tile_spacing:.3f} (overlap: {(1 - tile_spacing / self.base_size) * 100:.1f}%)")
        print(f"  - Pattern: {len(mixed_configs)} instances with varied transformations")
    
    def _recompute_normals_seamless(self, side='bottom', pixels=3):
        import numpy as np

        key_pos = 'my_shape_group.material_plane.vertex_positions'
        key_nrm = 'my_shape_group.material_plane.vertex_normals'

        v = np.array(self.params_template[key_pos], copy=True).astype(np.float32).reshape(-1, 3)
        G = int(round(np.sqrt(v.shape[0])))
        V = v.reshape(G, G, 3)

        p = int(pixels)
        if p < 0:
            p = 0
        if p > G - 1:
            p = G - 1

        Tj = np.empty_like(V, dtype=np.float32)
        Tj[:, 1:-1] = 0.5 * (V[:, 2:, :] - V[:, :-2, :])
        Tj[:, 0, :] = V[:, 1, :] - V[:, 0, :]
        Tj[:, -1, :] = V[:, -1, :] - V[:, -2, :]

        Ti = np.empty_like(V, dtype=np.float32)
        Ti[1:-1, :, :] = 0.5 * (V[2:, :, :] - V[:-2, :, :])
        Ti[0, :, :] = V[1, :, :] - V[0, :, :]
        Ti[-1, :, :] = V[-1, :, :] - V[-2, :, :]

        if side == 'bottom' and p > 0:
            ip = min(p, G - 1)  
            ip1 = min(p + 1, G - 1)  
            Ti[ip, :, :] = V[ip1, :, :] - V[ip, :, :]
            Ti[:p, :, :] = (V[ip1, :, :] - V[ip, :, :])[None, :, :]

        N = np.cross(Tj, Ti)
        L = np.linalg.norm(N, axis=-1, keepdims=True)
        N = N / np.maximum(L, 1e-8)

        if side == 'bottom' and p > 0:
            N[:p, :, :] = N[p, :, :]

        if float(N[:, :, 2].mean()) < 0.0:
            N = -N

        if key_nrm in self.params_template:
            self.params_template[key_nrm] = N.reshape(-1, 3).ravel()
            self.params_template.update()
        else:
            print("[warn] no 'vertex_normals'")

    def export_deformed_mesh(self, out_path: str):
        params = self.params_template
        v = np.array(params['my_shape_group.material_plane.vertex_positions'], copy=True).reshape(-1, 3)

        try:
            vn = np.array(params['my_shape_group.material_plane.vertex_normals'], copy=True).reshape(-1, 3)
            has_vn = True
        except Exception:
            vn = None
            has_vn = False

        grid = int(round(np.sqrt(v.shape[0])))

        tex_w = tex_h = grid
        eps_u = 0.5 / tex_w
        eps_v = 0.5 / tex_h

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("# Deformed mesh exported from StreamingBTFRenderer\n")
            for x, y, z in v:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            if has_vn:
                for nx, ny, nz in vn:
                    f.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
            for i in range(grid):
                for j in range(grid):
                    u = eps_u + j * (1 - 2 * eps_u) / (grid - 1)
                    t = eps_v + (grid - 1 - i) * (1 - 2 * eps_v) / (grid - 1)
                    f.write(f"vt {u:.8f} {t:.8f}\n")

            for i in range(grid - 1):
                for j in range(grid - 1):
                    v0 = i * grid + j + 1
                    v1 = v0 + 1
                    v2 = v0 + grid
                    v3 = v2 + 1
                    if has_vn:
                        f.write(f"f {v0}/{v0}/{v0} {v1}/{v1}/{v1} {v2}/{v2}/{v2}\n")
                        f.write(f"f {v1}/{v1}/{v1} {v3}/{v3}/{v3} {v2}/{v2}/{v2}\n")
                    else:
                        f.write(f"f {v0}/{v0} {v1}/{v1} {v2}/{v2}\n")
                        f.write(f"f {v1}/{v1} {v3}/{v3} {v2}/{v2}\n")

        print(f"[mesh] exported: {out_path}")
        return str(out_path)

    def update_material(self, material_data, rotation_angle=0, material_name=None):
        if self.scene_template is None:
            self._create_scene_template()
        self.params_template['my_shape_group.material_plane.bsdf.brdf_0.specular'] = 1
        
        rotation_steps = rotation_angle // 90
        
        if 'basecolor' in material_data:
            basecolor = material_data['basecolor']
            if isinstance(basecolor, torch.Tensor):
                if basecolor.shape[0] == 4:
                    print(f"  Converting RGBA to RGB (dropping alpha channel)")
                    basecolor = basecolor[:3]
                basecolor = basecolor.permute(1, 2, 0).cpu().numpy()
            
            if rotation_steps > 0:
                basecolor = np.rot90(basecolor, rotation_steps)
            
            basecolor = basecolor.astype(np.float32)
            basecolor = np.clip(basecolor, 0, 1)
            self.params_template['my_shape_group.material_plane.bsdf.brdf_0.base_color.data'] = basecolor

        if 'roughness' in material_data:
            roughness = material_data['roughness']
            if isinstance(roughness, torch.Tensor):
                if roughness.dim() == 2:
                    roughness = roughness.unsqueeze(0)
                roughness_np = roughness.cpu().numpy()
                if roughness_np.dtype == np.int16:
                    roughness_np = roughness_np.astype(np.float32)
                roughness_np = roughness_np.transpose(1, 2, 0)
            
            if rotation_steps > 0:
                roughness_np = np.rot90(roughness_np, rotation_steps)
            
            roughness_np = roughness_np.astype(np.float32)
            roughness_np = np.clip(roughness_np, 0.15, 1)
            self.params_template['my_shape_group.material_plane.bsdf.brdf_0.roughness.data'] = roughness_np
        
        if 'metallic' in material_data:
            metallic = material_data['metallic']
            if isinstance(metallic, torch.Tensor):
                metallic_np = metallic.permute(1, 2, 0).cpu().numpy()
            
            if rotation_steps > 0:
                metallic_np = np.rot90(metallic_np, rotation_steps)
            
            metallic_np = metallic_np.astype(np.float32)
            metallic_np = np.clip(metallic_np, 0, 1)
            self.params_template['my_shape_group.material_plane.bsdf.brdf_0.metallic.data'] = metallic_np
        
        if 'height' in material_data or 'displacement' in material_data:
            if self.cached_height_map is None:
                height = material_data.get('height')
                if isinstance(height, torch.Tensor):
                    if height.dim() == 2:
                        height = height.unsqueeze(0)
                    height_np = height[0].cpu().numpy()
                    if height_np.dtype == np.int16:
                        print(f"  Converting int16 height to float32")
                        height_np = height_np.astype(np.float32) / (256 * 256)
                
                if rotation_steps > 0:
                    height_np = np.rot90(height_np, rotation_steps)
                
                height_np = gaussian_filter(height_np, sigma=self.gaussian_sigma, mode='wrap')
                self.cached_height_map = height_np.copy() 
            else:
                print(f"  Using cached heightmap (rotation only)")
                height_np = np.rot90(self.cached_height_map, rotation_steps)
            
            self._apply_displacement(height_np, material_name=material_name)
            self.collapse_mesh_border(pixels=4, side='bottom', eps_frac=0.001, drop_z_frac=5)
        else:
            self.params_template['my_shape_group.material_plane.vertex_positions'] = dr.copy(self.original_vertices)
        
        self.params_template.update()

    def _apply_displacement(self, height_map, scale=4, reference_mode='mean', material_name=None):
        import numpy as np
        from scipy.ndimage import zoom

        v0 = np.array(self.original_vertices, copy=True).reshape(-1, 3).astype(np.float32)
        grid_size = self.basemeshsize

        if height_map.shape != (grid_size, grid_size):
            zoom_factor = (grid_size / height_map.shape[0], grid_size / height_map.shape[1])
            height_map = zoom(height_map, zoom_factor, order=1, mode='reflect')

        H = height_map.astype(np.float32, copy=False)

        if reference_mode == 'mean':
            ref_h = float(H.mean())
        elif reference_mode == 'min':
            ref_h = float(H.min())
        elif reference_mode == 'max':
            ref_h = float(H.max())
        elif reference_mode == 'center':
            ref_h = 0.5
        else:
            ref_h = float(H.mean())

        axis_idx = 2
        disp = (H - ref_h) * float(scale)
        v = v0.copy()
        v[:, axis_idx] = disp.reshape(-1)

        # 记录位移统计信息
        disp_min = float(disp.min())
        disp_max = float(disp.max())
        disp_range = disp_max - disp_min
        disp_mean = float(disp.mean())
        disp_std = float(disp.std())
        disp_abs_sum = float(np.abs(disp).sum())
        
        stats = {
            'disp_min': disp_min,
            'disp_max': disp_max,
            'disp_range': disp_range,
            'disp_mean': disp_mean,
            'disp_std': disp_std,
            'disp_abs_sum': disp_abs_sum,
            'scale': scale,
            'ref_height': ref_h
        }
        
        if material_name:
            self.displacement_stats[material_name] = stats
            print(f"  Displacement stats for {material_name}:")
            print(f"    Range: [{disp_min:.4f}, {disp_max:.4f}], Total range: {disp_range:.4f}")
            print(f"    Mean: {disp_mean:.4f}, Std: {disp_std:.4f}")

        self.params_template['my_shape_group.material_plane.vertex_positions'] = v.reshape(-1)
        self.params_template.update()
        
        return stats

    def save_displacement_stats_to_csv(self, output_path):
        """保存所有材料的位移统计到 CSV 文件"""
        output_path = Path(output_path)
        csv_path = output_path / 'displacement_stats1.csv'
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'material_name', 
                'disp_min', 
                'disp_max', 
                'disp_range', 
                'disp_mean', 
                'disp_std', 
                'disp_abs_sum',
                'scale',
                'ref_height'
            ])
            
            for material_name, stats in self.displacement_stats.items():
                writer.writerow([
                    material_name,
                    stats['disp_min'],
                    stats['disp_max'],
                    stats['disp_range'],
                    stats['disp_mean'],
                    stats['disp_std'],
                    stats['disp_abs_sum'],
                    stats['scale'],
                    stats['ref_height']
                ])
        
        print(f"Displacement stats saved to: {csv_path}")
        return csv_path
    def save_scene_state(self, output_path):
        state_keys = ['sensor.camera_position',
                'sensor.to_world',
                'area_light_sphere.to_world',
                'area_light_sphere.emitter.radiance.value',
                'light.position',
                'light.intensity.value',
                'sensor.x_fov',
                ]+[f'instance_mixed_{i}.to_world' for i in range(8)]
        state = {}
        for key in state_keys:       
            p = self.params_template.get(key)
            if p is None:
                state[key] = None
            elif 'to_world' in key:
                state[key] = p.matrix.numpy().tolist()
            else:
                state[key] = p.numpy()
        
        # Ensure parent directory exists (defensive against Lustre filesystem issues)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, state)

    def render_btf_sequence(self, output_path, total_frames=129, save_frames=False, rotation_angle=0, suffix="", save_exr=False, input_states=None, traj='sparse'):
        assert traj == 'sparse'
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        if save_frames:
            (output_path / 'png').mkdir(parents=True, exist_ok=True)
        if save_exr:
            (output_path / 'exr').mkdir(parents=True, exist_ok=True)
        if save_exr or save_frames:
            (output_path / 'scene_state').mkdir(parents=True, exist_ok=True)


        video_path = output_path / f'btf_sequence{suffix}.mp4'
        w = h = int(self.resolution)

        ff_proc = None
        try:
            ff_proc, encoder_name = self._open_ffmpeg(video_path, w, h, fps=30, prefer_nvenc=False, cq=0)
        except Exception as e:
            print(f"[FFmpeg] fail back to libx264 ...")
            ff_proc, encoder_name = self._open_ffmpeg(video_path, w, h, fps=30, prefer_nvenc=False, cq=0)

        phase1_frames = total_frames // 2
        phase3_frames = total_frames - phase1_frames

        frame_idx = 0
        camera_pos = [0.0, 0.0, self.camera_height+2]
        light_horizontal_radius = self.camera_radius#self.light_radius * np.sin(np.pi / 3)
        base_distance = self.lightgeight * 2+2

        # ---------- Phase 1: Light rotation ----------
        for i in range(phase1_frames):

            # print(f"\n=== Frame {i} (Light rotation) ===")
            # if i == 0:  
            #     sampler = self.scene_template.sampler()
            #     for test_idx in [0, 1000, 10000]:
            #         s = sampler.clone()
            #         s.seed(test_idx, 0)
            #         vals = [s.next_1d() for _ in range(3)]
            #         print(f"  Pixel {test_idx}: {vals}")
            t = i / (phase1_frames - 1) if phase1_frames > 1 else 0
            theta = 2 * np.pi * t

            light_pos = [
                light_horizontal_radius * np.cos(theta),
                light_horizontal_radius * np.sin(theta),
                self.lightgeight * 2+2
            ]
            actual_distance = np.sqrt(light_pos[0]**2 + light_pos[1]**2 + light_pos[2]**2)
            intensity_correction = (actual_distance / base_distance) ** 2
            corrected_intensity = [float(val * intensity_correction) for val in self.base_light_intensity]
            # print(camera_pos)
            if self.camera:
                self.params_template['sensor.camera_position'] = mi.Vector3f(
                    float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])
                )
            else: 
                # transformc = mi.Transform4f()
                # transformc = transformc.translate(mi.Point3f(float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])))
                # self.params_template['sensor.to_world'] = transformc
                self.params_template['sensor.to_world'] = mi.ScalarTransform4f().look_at(origin=[float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])],target=[0, 0, 0],up=[0,1 , 0])

            if self.arealight:
                transform = mi.Transform4f()
                transform = transform.translate(mi.Point3f(float(light_pos[0]), float(light_pos[1]), float(light_pos[2])))
                transform = transform.scale(self.area_light_radius)
                self.params_template['area_light_sphere.to_world'] = transform
                self.params_template['area_light_sphere.emitter.radiance.value'] = mi.Vector3f(*corrected_intensity)
            else:
                self.params_template['light.position'] = mi.Vector3f(
                    float(light_pos[0]), float(light_pos[1]), float(light_pos[2])
                )
                self.params_template['light.intensity.value'] = mi.Vector3f(*corrected_intensity)
            self.params_template['sensor.x_fov']=37
            self.params_template.update()

            if input_states is not None:
                load_scene_state(input_states[frame_idx], self.params_template, not self.arealight)

            seed = np.random.randint(0, 2**31 - 1)
            image_all = mi.render(self.scene_template, seed=seed,spp=16)

            # image_all = mi.render(self.scene_template)


            beauty  = image_all[..., 0:3]
            albedo  = image_all[..., 3:6]
            normals = image_all[..., 6:9]

            image = self.denoiser(beauty, albedo=albedo, normals=normals) if self.use_optix_denoiser else beauty
            if save_frames:
                frame_path = output_path / 'png' / f"btf_frame_{suffix}_{frame_idx:04d}.png"
                mi.util.write_bitmap(str(frame_path), image)

            if save_exr:
                exr_path = output_path / 'exr' / f"btf_frame_{suffix}_{frame_idx:04d}.exr"
                mi.util.write_bitmap(str(exr_path), image)

            if save_frames or save_exr:
                self.save_scene_state(output_path / 'scene_state' /f"btf_frame_{suffix}_{frame_idx:04d}.npy")

            srgb = self._linear_to_srgb_gpu(image)
            srgb_np = np.array(srgb)  
            rgb_uint8 = (srgb_np * 255.0 + 0.5).astype(np.uint8)
            try:
                ff_proc.stdin.write(rgb_uint8.tobytes())
            except (BrokenPipeError, OSError) as e:
                print(f"[FFmpeg] fail , using libx264 ...")
                try:
                    if ff_proc:
                        try:
                            ff_proc.stdin.close()
                        except Exception:
                            pass
                        ff_proc.wait(timeout=2)
                except Exception:
                    pass
                ff_proc, encoder_name = self._open_ffmpeg(video_path, w, h, fps=30, prefer_nvenc=False, cq=0)
                ff_proc.stdin.write(rgb_uint8.tobytes())



            frame_idx += 1

        # ---------- Phase 2: Static light, camera rotation ----------
        light_pos = [0.0, 0.0, self.lightgeight * 2+2]

        for i in range(phase3_frames):
            t = i / (phase3_frames - 1) if phase3_frames > 1 else 0
            theta = 2 * np.pi * t
            camera_pos = [
                self.camera_radius * np.cos(theta),
                -self.camera_radius * np.sin(theta),
                self.camera_height+2
            ]

            # self.params_template['sensor.camera_position'] = mi.Vector3f(
            #     float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])
            # )
            # print(camera_pos)
            if self.camera:
                self.params_template['sensor.camera_position'] = mi.Vector3f(
                    float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])
                )
            else: 
                # transformc = mi.Transform4f()
                # transformc = transformc.translate(mi.Point3f(float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])))
                # self.params_template['sensor.to_world'] = transformc
                self.params_template['sensor.to_world'] = mi.ScalarTransform4f().look_at(origin=[float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])],target=[0, 0, 0],up=[0, 1, 0])


            if self.arealight:
                transform = mi.Transform4f()
                transform = transform.translate(mi.Point3f(float(light_pos[0]), float(light_pos[1]), float(light_pos[2])))
                transform = transform.scale(self.area_light_radius)
                self.params_template['area_light_sphere.to_world'] = transform
                self.params_template['area_light_sphere.emitter.radiance.value'] = mi.Vector3f(*self.base_light_intensity)
            else:
                self.params_template['light.position'] = mi.Vector3f(
                    float(light_pos[0]), float(light_pos[1]), float(light_pos[2])
                )
                self.params_template['light.intensity.value'] = mi.Vector3f(*self.base_light_intensity)

            self.params_template['sensor.x_fov']=30

            self.params_template.update()
            if input_states is not None:
                load_scene_state(input_states[frame_idx], self.params_template, not self.arealight)

            seed = np.random.randint(0, 2**31 - 1)
            image_all = mi.render(self.scene_template, seed=seed,spp=16)
            beauty  = image_all[..., 0:3]
            albedo  = image_all[..., 3:6]
            normals = image_all[..., 6:9]

            image = self.denoiser(beauty, albedo=albedo, normals=normals) if self.use_optix_denoiser else beauty
            
            if save_frames:
                frame_path = output_path / 'png' / f"btf_frame_{suffix}_{frame_idx:04d}.png"
                mi.util.write_bitmap(str(frame_path), image)

            if save_exr:
                exr_path = output_path / 'exr' / f"btf_frame_{suffix}_{frame_idx:04d}.exr"
                mi.util.write_bitmap(str(exr_path), image)

            if save_frames or save_exr:
                self.save_scene_state(output_path / 'scene_state' /f"btf_frame_{suffix}_{frame_idx:04d}.npy")

            srgb = self._linear_to_srgb_gpu(image)
            srgb_np = np.array(srgb)
            rgb_uint8 = (srgb_np * 255.0 + 0.5).astype(np.uint8)
            ff_proc.stdin.write(rgb_uint8.tobytes())
 

            frame_idx += 1

        try:
            ff_proc.stdin.close()
        finally:
            ff_proc.wait()
        print(f"Video saved: {video_path}")
        return video_path
    def render_btf_sequence_validation(self, output_path, total_frames=81, save_frames=False, 
                            rotation_angle=0, suffix="", save_exr=False, input_states=None, traj='dense'):
        """使用 validation_cameras 轨迹渲染 BTF 序列，均匀采样到指定帧数"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        if save_frames:
            (output_path / 'png').mkdir(parents=True, exist_ok=True)
        if save_exr:
            (output_path / 'exr').mkdir(parents=True, exist_ok=True)
        if save_exr or save_frames:
            (output_path / 'scene_state').mkdir(parents=True, exist_ok=True)


        video_path = output_path / f'btf_sequence{suffix}.mp4'
        w = h = int(self.resolution)

        # 生成完整的 validation camera 轨迹并均匀采样
        function_name = dense_cameras if traj == 'dense' else validation_cameras if traj == 'grazing' else None
        full_states = function_name(
            total_n_frames=81,
            area_light_radius=self.area_light_radius,
            lightgeight=self.lightgeight,
            camera_height=self.camera_height,
            base_light_intensity=self.base_light_intensity,
            camera_radius=self.camera_radius,
            use_area_light=self.arealight
        )
        val_states = full_states
        n_frames = len(full_states)
        
        print(f"Rendering {n_frames} frames using validation camera sequence (sampled from 320)")

        ff_proc = None
        try:
            ff_proc, encoder_name = self._open_ffmpeg(video_path, w, h, fps=5, prefer_nvenc=False, cq=0)
        except Exception as e:
            print(f"[FFmpeg] fail back to libx264 ...")
            ff_proc, encoder_name = self._open_ffmpeg(video_path, w, h, fps=5, prefer_nvenc=False, cq=0)

        for frame_idx in range(n_frames):
            state = val_states[frame_idx]
            
            # 获取原始位置数据
            camera_pos = state['camera_pos']
            camera_up = state['camera_up']
            light_pos = state['light_pos']
            light_intensity = state['light_intensity']
            fov = state.get('sensor.x_fov', 30)
            
            # 设置相机 - 和原始代码一样的方式
            self.params_template['sensor.to_world'] = mi.ScalarTransform4f().look_at(
                origin=[float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])],
                target=[0, 0, 0],
                up=[float(camera_up[0]), float(camera_up[1]), float(camera_up[2])]
            )
            self.params_template['sensor.x_fov'] = fov
            
            # 设置光源 - 和原始代码一样的方式
            if self.arealight:
                transform = mi.Transform4f()
                transform = transform.translate(mi.Point3f(float(light_pos[0]), float(light_pos[1]), float(light_pos[2])))
                transform = transform.scale(self.area_light_radius)
                self.params_template['area_light_sphere.to_world'] = transform
                self.params_template['area_light_sphere.emitter.radiance.value'] = mi.Vector3f(*light_intensity)
            else:
                self.params_template['light.position'] = mi.Vector3f(
                    float(light_pos[0]), float(light_pos[1]), float(light_pos[2])
                )
                self.params_template['light.intensity.value'] = mi.Vector3f(*light_intensity)

            self.params_template.update()

            # 渲染
            seed = np.random.randint(0, 2**31 - 1)
            image_all = mi.render(self.scene_template, seed=seed, spp=16)

            beauty = image_all[..., 0:3]
            albedo = image_all[..., 3:6]
            normals = image_all[..., 6:9]

            image = self.denoiser(beauty, albedo=albedo, normals=normals) if self.use_optix_denoiser else beauty

            if save_frames:
                frame_path = output_path / 'png' / f"btf_frame_{suffix}_{frame_idx:04d}.png"
                mi.util.write_bitmap(str(frame_path), image)

            if save_exr:
                exr_path = output_path / 'exr' / f"btf_frame_{suffix}_{frame_idx:04d}.exr"
                mi.util.write_bitmap(str(exr_path), image)

            if save_frames or save_exr:
                self.save_scene_state(output_path / 'scene_state' /f"btf_frame_{suffix}_{frame_idx:04d}.npy")


            srgb = self._linear_to_srgb_gpu(image)
            srgb_np = np.array(srgb)
            rgb_uint8 = (srgb_np * 255.0 + 0.5).astype(np.uint8)
            
            try:
                ff_proc.stdin.write(rgb_uint8.tobytes())
            except (BrokenPipeError, OSError) as e:
                print(f"[FFmpeg] fail, using libx264 ...")
                try:
                    if ff_proc:
                        ff_proc.stdin.close()
                        ff_proc.wait(timeout=2)
                except Exception:
                    pass
                ff_proc, encoder_name = self._open_ffmpeg(video_path, w, h, fps=30, prefer_nvenc=False, cq=0)
                ff_proc.stdin.write(rgb_uint8.tobytes())

            if frame_idx % 10 == 0:
                print(f"  Frame {frame_idx + 1}/{n_frames}")

        try:
            ff_proc.stdin.close()
        finally:
            ff_proc.wait()
        
        print(f"Video saved: {video_path}")
        return video_path

    def linear_to_srgb(self, x):
        x = np.clip(x, 0.0, 1.0)
        a = 0.055
        return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * np.power(x, 1 / 2.4) - a)
            
    def _create_video_from_frames(self, frames, video_path, fps=30):
        print(f"Creating video: {video_path}")
        
        frames_uint8 = []
        for frame in frames:
            lin = np.clip(frame, 0, None)
            srgb = self.linear_to_srgb(lin)
            frame_uint8 = (srgb * 255.0 + 0.5).astype(np.uint8)
            frames_uint8.append(frame_uint8)
                
        height, width = frames_uint8[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        for frame in frames_uint8:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Video saved: {video_path}")
        
    def process_local_dataset(self, parquet_dir, output_base_dir, max_materials=None, 
                              start_idx=0, save_exr=False, process_mode='all', 
                              total_frames=81, save_frames=False, traj='sparse'):
        output_base = Path(output_base_dir)
        output_base.mkdir(parents=True, exist_ok=True)
        parquet_dir = Path(parquet_dir)
        input_states=[np.load(f'data/matsynth_scene_state/81 frames-sub129/btf_frame__{i:04d}.npy', allow_pickle=True).item() for i in range(81)]


        print(f"Loading dataset from: {parquet_dir}")
        ds = load_dataset(
            "gvecchio/MatSynth",
            split="train",
        )
        material_count = 0
        current_idx = 0
        processed_materials = []  # Track for metadata.csv

        print("total materials:", len(ds))

        repeated_names = {'Leather 003': [1303, 1365], 'Leather 014': [1102, 1390], 'Leather 013': [1, 1940], 'Leather 015': [1643, 1987], 'Asphalt 01': [987, 2014], 'Marble 01': [167, 2285], 'Leather 022': [938, 2496], 'Brick Wall 04': [2053, 2671], 'Leather 019': [1175, 2728], 'Leather 018': [1730, 2788], 'Leather 021': [1336, 3026], 'Leather 012': [419, 3206], 'Wood 04': [1158, 3299], 'Leather 005': [559, 3520], 'Brick Wall 21': [1858, 3769], 'Leather 017': [2074, 3860], 'Brick Wall 22': [183, 3982], 'Leather 002': [2399, 4428], 'Leather 020': [1950, 4429], 'Metal Tiles 02': [359, 4584], 'Leather 006': [2677, 4726], 'Leather 010': [4856, 4962], 'Leather 004': [250, 5040], 'Leather 011': [3274, 5126], 'Metal Tiles 01': [3247, 5299], 'Leather 009': [4689, 5448]}
        repeated_names = {k:v[1] for k,v in repeated_names.items()}

        mapping_repeated_names= {
    'Asphalt 01': 'Asphalt 01_987',
    'Brick Wall 04': 'Brick Wall 04_2053',
    'Brick Wall 21': 'Brick Wall 21_3769',
    'Brick Wall 22': 'Brick Wall 22_183',
    'Leather 002': 'Leather 002_2399',
    'Leather 003': 'Leather 003_1303',
    'Leather 004': 'Leather 004_250',
    'Leather 005': 'Leather 005_559',
    'Leather 006': 'Leather 006_2677',
    'Leather 009': 'Leather 009_4689',
    'Leather 010': 'Leather 010_4856',
    'Leather 011': 'Leather 011_3274',
    'Leather 012': 'Leather 012_3206',
    'Leather 013': 'Leather 013_1',
    'Leather 014': 'Leather 014_1102',
    'Leather 015': 'Leather 015_1643',
    'Leather 017': 'Leather 017_3860',
    'Leather 018': 'Leather 018_1730',
    'Leather 019': 'Leather 019_1175',
    'Leather 020': 'Leather 020_4429',
    'Leather 021': 'Leather 021_3026',
    'Leather 022': 'Leather 022_938',
    'Marble 01': 'Marble 01_167',
    'Metal Tiles 01': 'Metal Tiles 01_3247',
    'Metal Tiles 02': 'Metal Tiles 02_359',
    'Wood 04': 'Wood 04_3299'
}

        ds = ds.skip(start_idx)

        # skipped_ds = ds.skip(start_idx)
        # print(f"Skipped {start_idx} materials")
        current_idx = start_idx


        for material_data in ds:
            if current_idx < start_idx:
                current_idx += 1
                print(f"Skipping material {current_idx:04d} to reach start_idx {start_idx}")
                continue

            should_process = False
            if process_mode == 0:
                should_process = True
            elif process_mode == 1:
                should_process = current_idx % 2 == 1
            elif process_mode == 2:
                should_process = current_idx % 2 == 0
            else:
                should_process = True
            
            if not should_process:
                current_idx += 1
                continue

            skip_material = False
            # for key in ['basecolor', 'roughness', 'metallic', 'height']:
            #     if key in material_data and material_data[key] is not None:
            #         texture = material_data[key]
            #         if hasattr(texture, 'size'):
            #             width, height = texture.size
            #             if width != 4096 or height != 4096:
            #                 print(f"  Skipping material {material_data['metadata'].get('name', f'material_{current_idx:04d}')} - resolution is {width}x{height}, not 4096x4096")
            #                 skip_material = True
            #                 break
            #         elif isinstance(texture, torch.Tensor):
            #             h, w = texture.shape[-2:]
            #             if h != 4096 or w != 4096:
            #                 print(f"  Skipping material {material_data['metadata'].get('name', f'material_{current_idx:04d}')} - resolution is {h}x{w}, not 4096x4096")
            #                 skip_material = True
            #                 break
            
            if skip_material:
                current_idx += 1
                material_count += 1
                continue

            if max_materials and material_count >= max_materials:
                break

            material_name = material_data['metadata'].get('name', f'material_{current_idx:04d}')
            height_factor = material_data['metadata'].get('height_factor')
            height_mean = material_data['metadata'].get('height_mean')
            if material_name in repeated_names:
                material_name = f'{material_name}_{current_idx}'
                print(f"Duplicate material name: {material_name}")


            print(f"\nProcessing [{current_idx + 1}]: {material_name}")

            output_path = output_base / material_name
            all_exist = True
            for angle in [0]:
                suffix = f"_{angle}deg" if angle > 0 else ""
                if not (output_path / f'btf_sequence{suffix}.mp4').exists():
                    all_exist = False
                    break
            
            if all_exist:
                print(f"Error: Skipping - already processed (all rotations)")
                current_idx += 1
                material_count += 1
                continue

            processed_data = {}
            for key in ['basecolor', 'roughness', 'metallic', 'height', 'normal']:
                if key in material_data and material_data[key] is not None:
                    texture = material_data[key]
                    
                    if hasattr(texture, 'mode'):
                        if texture.mode == 'I':
                            texture_np = np.array(texture, dtype=np.uint16)
                            texture = torch.from_numpy(texture_np).float() / 65535.0
                            texture = texture.unsqueeze(0)
                        elif texture.mode == 'I;16':
                            texture_np = np.array(texture, dtype=np.uint16)
                            texture = torch.from_numpy(texture_np).float() / 65535.0
                            texture = texture.unsqueeze(0)
                        else:
                            texture = TF.to_tensor(texture)
                    else:
                        if texture.dtype == torch.int16:
                            texture = texture.to(torch.int32) + 32768
                            texture = texture.float() / 65535.0
                        elif not isinstance(texture, torch.Tensor):
                            texture = torch.tensor(np.array(texture))
                    
                    if key == 'height':
                        texture = (texture - height_mean) * height_factor

                    if texture.shape[-2:] != (self.resolution, self.resolution) and key != 'height':
                        if texture.dim() == 2:
                            texture = texture.unsqueeze(0)
                        texture = TF.resize(texture, (self.resolution, self.resolution))
                    
                    processed_data[key] = texture
                    
            self.cached_height_map = None
            for i, angle in enumerate([0]):
                self.update_material(processed_data, rotation_angle=angle, material_name=material_name)
                
                suffix = f"_{angle}deg" if angle > 0 else ""
                print(f"  Rendering rotation {angle} degrees...")
                from time import time
                start_time = time()
                function_name = self.render_btf_sequence if traj == 'sparse' else self.render_btf_sequence_validation
                video_path = function_name(
                    output_path, 
                    total_frames=total_frames,
                    rotation_angle=angle, 
                    suffix=suffix, 
                    save_exr=save_exr,
                    save_frames=save_frames,
                    input_states=input_states,
                    traj=traj
                )
                end_time = time()
                print(f"  Time taken: {end_time - start_time} seconds")
                # start_time = time()
                # video_path = self.render_btf_sequence_validation(
                #     output_path, 
                #     total_frames=total_frames,
                #     rotation_angle=angle, 
                #     suffix=suffix, 
                #     save_exr=save_exr,
                #     save_frames=save_frames
                # )
                # end_time = time()
                # print(f"  Time taken: {end_time - start_time} seconds")
                print(f"  Completed: {video_path}")

            processed_materials.append(material_name)
            material_count += 1
            current_idx += 1

            del processed_data
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 保存位移统计到 CSV
        self.save_displacement_stats_to_csv(output_base)
        
        # Save metadata.csv
        metadata_csv = output_base / 'metadata_clean.csv'
        with open(metadata_csv, 'w') as f:
            f.write('video,prompt\n')
            for name in processed_materials:
                f.write(f'"{name}/btf_sequence.mp4",""\n')
        print(f"Saved metadata to: {metadata_csv}")
        
        print(f"\nProcessed {material_count} materials")

    def _compute_normals_rows_subset(self, V, i0, i1):
        import numpy as np
        Gy, Gx, _ = V.shape
        i0 = max(0, int(i0))
        i1 = min(Gy - 1, int(i1))

        V_row = V[i0:i1 + 1]

        Tj = np.empty_like(V_row, dtype=np.float32)
        Tj[:, 1:-1] = 0.5 * (V_row[:, 2:] - V_row[:, :-2])
        Tj[:, 0] = V_row[:, 1] - V_row[:, 0]
        Tj[:, -1] = V_row[:, -1] - V_row[:, -2]

        Ti = np.empty_like(V_row, dtype=np.float32)
        idx = np.arange(i0, i1 + 1)

        mid_mask = (idx > 0) & (idx < Gy - 1)
        if np.any(mid_mask):
            im = idx[mid_mask]
            Ti[mid_mask] = 0.5 * (V[im + 1] - V[im - 1])

        for k, i in enumerate(idx):
            if i == 0:
                Ti[k] = V[1] - V[0]
            elif i == Gy - 1:
                Ti[k] = V[Gy - 1] - V[Gy - 2]

        N = np.cross(Tj, Ti)
        L = np.linalg.norm(N, axis=-1, keepdims=True)
        N = N / np.maximum(L, 1e-8)
        return N.astype(np.float32)

    def _compute_normals_cols_subset(self, V, j0, j1):
        import numpy as np
        Gy, Gx, _ = V.shape
        j0 = max(0, int(j0))
        j1 = min(Gx - 1, int(j1))
        js = np.arange(j0, j1 + 1)
        cnt = len(js)

        Tj = np.empty((Gy, cnt, 3), dtype=np.float32)
        for k, j in enumerate(js):
            if j == 0:
                Tj[:, k] = V[:, 1] - V[:, 0]
            elif j == Gx - 1:
                Tj[:, k] = V[:, Gx - 1] - V[:, Gx - 2]
            else:
                Tj[:, k] = 0.5 * (V[:, j + 1] - V[:, j - 1])

        Ti = np.empty((Gy, cnt, 3), dtype=np.float32)
        for k, j in enumerate(js):
            Ti[1:Gy - 1, k] = 0.5 * (V[2:, j] - V[:-2, j])
            Ti[0, k] = V[1, j] - V[0, j]
            Ti[Gy - 1, k] = V[Gy - 1, j] - V[Gy - 2, j]

        N = np.cross(Tj, Ti)
        L = np.linalg.norm(N, axis=-1, keepdims=True)
        N = N / np.maximum(L, 1e-8)
        return N.astype(np.float32)

    def collapse_mesh_border(self, pixels=3, side='bottom', eps_frac=0.1, drop_z_frac=0.5):
        key = 'my_shape_group.material_plane.vertex_positions'
        v = np.array(self.params_template[key], copy=True).astype(np.float32)
        n_verts = v.size // 3

        if hasattr(self, 'grid_nx') and hasattr(self, 'grid_ny') and self.grid_nx * self.grid_ny == n_verts:
            Gx, Gy = int(self.grid_nx), int(self.grid_ny)
        else:
            G = int(round((n_verts) ** 0.5))
            Gx = Gy = G
            self.grid_nx, self.grid_ny = Gx, Gy

        V = v.reshape(Gy, Gx, 3)

        px_x = self.base_size / max(self.grid_nx - 1, 1)
        px_y = self.base_size / max(self.grid_ny - 1, 1)

        if side in ('bottom', 'top'):
            pixel_world = px_y
            band_thick = eps_frac * pixel_world
            drop_z_unit = drop_z_frac * pixel_world
        else:
            pixel_world = px_x
            band_thick = eps_frac * pixel_world
            drop_z_unit = drop_z_frac * pixel_world

        if side == 'bottom':
            y_ref = V[pixels, :, 1].copy()
            for k in range(pixels):
                y_new = y_ref - band_thick * (pixels - k)
                V[k, :, 1] = y_new
                V[k, :, 2] -= drop_z_unit * (pixels - k)
        elif side == 'top':
            y_ref = V[-pixels - 1, :, 1].copy()
            for k in range(pixels):
                y_new = y_ref + band_thick * (pixels - k)
                V[-1 - k, :, 1] = y_new
                V[-1 - k, :, 2] -= drop_z_unit * (pixels - k)
        elif side == 'left':
            x_ref = V[:, pixels, 0].copy()
            for k in range(pixels):
                x_new = x_ref - band_thick * (pixels - k)
                V[:, k, 0] = x_new
                V[:, k, 2] -= drop_z_unit * (pixels - k)
        elif side == 'right':
            x_ref = V[:, -pixels - 1, 0].copy()
            for k in range(pixels):
                x_new = x_ref + band_thick * (pixels - k)
                V[:, -1 - k, 0] = x_new
                V[:, -1 - k, 2] -= drop_z_unit * (pixels - k)
        else:
            raise ValueError("side must be bottom/top/left/right")

        self.params_template[key] = V.reshape(-1)
        self.params_template.update()

        self._collapsed = {'side': side, 'pixels': int(pixels), 'band_thick': float(band_thick)}
        self._recompute_normals_seamless(side=side, pixels=pixels)

    def retile_instances_to_bbox(self, ignore_collapsed_band=True):
        import numpy as np
        import math
        import mitsuba as mi

        v = np.array(self.params_template['my_shape_group.material_plane.vertex_positions'], copy=False).astype(np.float32)

        n_verts = v.size // 3
        if hasattr(self, 'grid_nx') and hasattr(self, 'grid_ny') and self.grid_nx * self.grid_ny == n_verts:
            Gx, Gy = int(self.grid_nx), int(self.grid_ny)
            V = v.reshape(Gy, Gx, 3)

            span_x = float(V[:, :, 0].max() - V[:, :, 0].min())
            span_y = float(V[:, :, 1].max() - V[:, :, 1].min())

            if ignore_collapsed_band and hasattr(self, '_collapsed') and self._collapsed:
                side = self._collapsed['side']
                p = self._collapsed['pixels']
                if side == 'bottom':
                    y_min_vis = float(V[p, :, 1].min())
                    y_max_vis = float(V[-1, :, 1].max())
                    span_y = y_max_vis - y_min_vis
                elif side == 'top':
                    y_min_vis = float(V[0, :, 1].min())
                    y_max_vis = float(V[-p - 1, :, 1].max())
                    span_y = y_max_vis - y_min_vis
                elif side == 'left':
                    x_min_vis = float(V[:, p, 0].min())
                    x_max_vis = float(V[:, -1, 0].max())
                    span_x = x_max_vis - x_min_vis
                elif side == 'right':
                    x_min_vis = float(V[:, 0, 0].min())
                    x_max_vis = float(V[:, -p - 1, 0].max())
                    span_x = x_max_vis - x_min_vis
        else:
            V = v.reshape(-1, 3)
            span_x = float(V[:, 0].max() - V[:, 0].min())
            span_y = float(V[:, 1].max() - V[:, 1].min())

        tile_spacing_x = span_x
        tile_spacing_y = span_y

        mixed_configs = [
            (0, 0, 1, 1, 0),
            (1, 0, -1, 1, 0),
            (-1, 0, -1, 1, 0),
            (0, 1, 1, -1, 0),
            (0, -1, 1, -1, 0),
            (1, 1, 1, 1, 180),
            (-1, 1, 1, 1, 180),
            (1, -1, 1, 1, 180),
            (-1, -1, 1, 1, 180),
        ]

        for idx, (i, j, scale_x, scale_y, angle) in enumerate(mixed_configs):
            offset_x = i * tile_spacing_x
            offset_y = j * tile_spacing_y

            ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            ox, oy = (ca * offset_x + sa * offset_y, -sa * offset_x + ca * offset_y)
            if scale_x != 0:
                ox /= scale_x
            if scale_y != 0:
                oy /= scale_y

            T = mi.ScalarTransform4f()
            T = T.scale([scale_x, scale_y, 1])
            if angle != 0:
                T = T.rotate([0, 0, 1], angle)
            T = T.translate([ox, oy, 0])

            key = f'instance_mixed_{idx}.to_world'
            if key in self.params_template:
                self.params_template[key] = T

        self.params_template.update()

    def _ffmpeg_supports_encoder(self, name: str) -> bool:
        import subprocess
        import shutil
        if not shutil.which('ffmpeg'):
            raise RuntimeError("no ffmpeg")
        try:
            p = subprocess.run(
                ['ffmpeg', '-v', 'error', '-h', f'encoder={name}'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return p.returncode == 0
        except Exception:
            return False

    def _open_ffmpeg(self, path, w, h, fps=30, prefer_nvenc=True, cq=0):
        import subprocess
        import shutil
        if not shutil.which('ffmpeg'):
            raise RuntimeError("no ffmpeg")

        use_nvenc = prefer_nvenc and self._ffmpeg_supports_encoder('h264_nvenc')

        if use_nvenc:
            encoder = 'h264_nvenc'
            codec = ['-c:v', 'h264_nvenc', '-preset', 'p5', '-rc', 'vbr', '-cq', str(cq)]
        else:
            encoder = 'libx264'
            codec = ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', str(cq)]

        cmd = [
            'ffmpeg', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{w}x{h}',
            '-r', str(fps),
            '-i', '-',
            '-an',
            '-pix_fmt', 'yuv420p',
            *codec,
            str(path)
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        print(f"[FFmpeg] using: {encoder}")
        return proc, encoder

    def _write_frame_ffmpeg(self, proc, rgb_uint8):
        proc.stdin.write(rgb_uint8.tobytes())

    def _close_ffmpeg(self, proc):
        proc.stdin.close()
        proc.wait()


def process_materials_worker(gpu_id, indices, parquet_dir, output_dir, resolution, total_gpus, save_exr=False, total_frames=81, save_frames=False):
    print(f"\nWorker on GPU {gpu_id} starting, processing {len(indices)} materials")
    
    renderer = StreamingBTFRenderer(resolution=resolution, gpu_id=gpu_id)
    
    renderer.process_local_dataset(
        parquet_dir=parquet_dir,
        output_base_dir=output_dir,
        max_materials=len(indices),
        start_idx=indices[0],
        save_exr=save_exr,
        total_frames=total_frames,
        save_frames=save_frames
    )
    
    print(f"\nWorker on GPU {gpu_id} completed")


def parse_args():
    """Parse command line arguments using argparse"""
    parser = argparse.ArgumentParser(
        description='BTF Material Renderer - Render BTF sequences from material datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output paths
    parser.add_argument(
        '--input', '-i',
        type=str,
        default="D:\\vedio\\MatSynth\\data",
        help='Path to parquet dataset directory'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default="./btf_dataset_output26",
        help='Output directory for rendered videos'
    )

    parser.add_argument(
        '--traj', '-t',
        type=str,
        default="sparse",
        help='Trajectories to use for rendering'
    )

    
    # Processing range
    parser.add_argument(
        '--max-materials', '-n',
        type=int,
        default=None,
        help='Maximum number of materials to process (None for all)'
    )
    parser.add_argument(
        '--start-idx', '-s',
        type=int,
        default=0,
        help='Starting index in the dataset'
    )
    
    # GPU settings
    parser.add_argument(
        '--num-gpus', '-g',
        type=int,
        default=1,
        help='Number of GPUs to use for parallel processing'
    )
    
    # Rendering settings
    parser.add_argument(
        '--resolution', '-r',
        type=int,
        default=2048,
        help='Render resolution (width and height)'
    )
    parser.add_argument(
        '--total-frames', '-f',
        type=int,
        default=81,
        help='Total number of frames per video'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=15,
        help='Video frame rate'
    )
    
    # Output options
    parser.add_argument(
        '--save-exr',
        action='store_true',
        help='Save individual frames as EXR files (high disk usage warning)'
    )
    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='Save individual frames as PNG files'
    )
    
    # Processing mode
    parser.add_argument(
        '--process-mode', '-m',
        type=int,
        default=0,
        choices=[0, 1, 2],
        help='Processing mode: 0=all materials, 1=odd indices only, 2=even indices only'
    )
    
    # Denoiser
    parser.add_argument(
        '--no-denoiser',
        action='store_true',
        help='Disable OptiX denoiser'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    with dr.suspend_grad():
        print("=" * 60)
        print("BTF Material Renderer")
        print("=" * 60)
        print(f"Input directory:  {args.input}")
        print(f"Output directory: {args.output}")
        print(f"Max materials:    {args.max_materials if args.max_materials else 'All'}")
        print(f"Starting index:   {args.start_idx}")
        print(f"Number of GPUs:   {args.num_gpus}")
        print(f"Resolution:       {args.resolution}x{args.resolution}")
        print(f"Total frames:     {args.total_frames}")
        print(f"Save EXR:         {args.save_exr}")
        print(f"Process mode:     {args.process_mode}")
        print(f"OptiX denoiser:   {not args.no_denoiser}")
        print(f"Traj:   {args.traj}")
        print("=" * 60)
        
        if args.save_exr:
            print(f"  WARNING: Saving all frames as EXR will use significant disk space!")
            print(f"  Estimated: ~{args.resolution * args.resolution * 3 * 4 * args.total_frames / (1024**2):.1f} MB per material rotation")
        
        if args.num_gpus > 1 and torch.cuda.device_count() >= args.num_gpus:
            print(f"\nMulti-GPU mode: Using {args.num_gpus} GPUs")
            
            if args.max_materials:
                materials_per_gpu = args.max_materials // args.num_gpus
                remainder = args.max_materials % args.num_gpus
            else:
                print("Warning: Multi-GPU mode without max_materials may not balance load properly")
                materials_per_gpu = 1000
                remainder = 0
            
            processes = []
            for gpu_id in range(args.num_gpus):
                gpu_start_idx = args.start_idx + gpu_id * materials_per_gpu
                gpu_materials = materials_per_gpu + (1 if gpu_id < remainder else 0)
                
                if args.max_materials:
                    indices = list(range(gpu_start_idx, min(gpu_start_idx + gpu_materials, args.start_idx + args.max_materials)))
                else:
                    indices = list(range(gpu_start_idx, gpu_start_idx + materials_per_gpu))
                
                p = mp.Process(
                    target=process_materials_worker,
                    args=(gpu_id, indices, args.input, args.output, args.resolution, args.num_gpus, args.save_exr, args.total_frames, args.save_frames)
                )
                p.start()
                processes.append(p)
            
            for p in processes:
                p.join()
            
            print("\nAll GPU workers completed")
            if args.save_exr:
                print("Note: All frames have been saved as EXR files")
        else:
            print(f"\nSingle GPU mode")
            print(f"OptiX denoiser: {'Enabled' if not args.no_denoiser else 'Disabled'}")
            
            renderer = StreamingBTFRenderer(resolution=args.resolution)
            renderer.use_optix_denoiser = not args.no_denoiser
            
            renderer.process_local_dataset(
                parquet_dir=args.input,
                output_base_dir=args.output,
                max_materials=args.max_materials,
                start_idx=args.start_idx,
                save_exr=args.save_exr,
                process_mode=args.process_mode,
                total_frames=args.total_frames,
                save_frames=args.save_frames,
                traj=args.traj
            )
            
            if args.save_exr:
                print("Note: All frames have been saved as EXR files")


if __name__ == "__main__":
    main()