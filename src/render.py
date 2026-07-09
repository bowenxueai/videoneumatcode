# bowen's code

from tracemalloc import start
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
from integrators.custom_aov import ArbitraryIntegrator

from utils import load_scene_state, ensure_RGB_dims

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    from btf_sensor import BTFSensor
    print("BTF sensor loaded successfully")
except ImportError as e:
    print(f"Error: Could not import BTF sensor")
    raise
#mitsuba 3.6.4
mi.set_variant('cuda_ad_rgb')
# mi.set_variant('scalar_rgb')

if not hasattr(mi, '_clamped_path_registered'):
    class ClampedPath(mi.SamplingIntegrator):
        def __init__(self, props):
            super().__init__(props)
            self.clamp = props.get("radiance_clamp", 10.0)
            # path integrator
            self.base = mi.load_dict({
                "type": "path",
                "max_depth": props.get("max_depth", 6),
                "hide_emitters": props.get("hide_emitters", True)
            })
        
        def sample(self, scene, sampler, ray, medium=None, active=True):
           
            spec, mask, *aovs = self.base.sample(scene, sampler, ray, medium, active)
            # m = dr.max(spec)
            # print("max(spec)", m,"manmanam",dr.max(m))
            
            spec_clamped = dr.clamp(spec, 0.0, self.clamp)
            
            return (spec_clamped, mask, *aovs)
        
        def aov_names(self):
            return self.base.aov_names() if hasattr(self.base, 'aov_names') else []
        
        def traverse(self, callback):
            super().traverse(callback)
            callback.put_parameter('radiance_clamp', self.clamp, mi.ParamFlags.NonDifferentiable)
    
    mi.register_integrator("clamped_path", lambda p: ClampedPath(p))
    mi._clamped_path_registered = True
    print("Clamped path integrator registered successfully")


class StreamingBTFRenderer:

    def __init__(self, base_mesh_path=None, resolution=1024, base_size=4.0, gpu_id=0, grid_size = 1024, only_center_mesh_instance = False, arealight = True):
        self.cached_height_map = None 
        self.resolution = resolution
        self.base_size = base_size
        self.gpu_id = gpu_id
        self.calibration_output_dir = "./btf_dataset_output7"
        self.arealight=arealight
        self.camera_transform = mi.ScalarTransform4f().look_at(origin=[0, 0, 5.5],target=[0, 0, 0],up=[0, 1, 0])
        self.grid_size= grid_size
        self.default_disp_scale = 1
        mi.set_variant(f'cuda_ad_rgb')
        # mi.set_variant('scalar_rgb')
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
        
        print(f"Initializing BTF renderer on GPU {gpu_id} with resolution: {resolution}x{resolution}")
        print(f"Estimated texture memory per material: {resolution*resolution*3*4*5/(1024**3):.2f} GB")
        self.basemeshsize=1024
        if base_mesh_path is None:
            self.base_mesh_path = self._create_base_mesh(grid_size=self.grid_size)
        else:
            self.base_mesh_path = base_mesh_path

        self.scene_template = None
        self.params_template = None
        self.original_vertices = None
        self.camera = False        
        self.camera_radius = 4.0
        self.light_radius = 2.0
        self.camera_height = 4#4.0
        self.lightgeight=2
        
        self.gaussian_sigma = 0.5
        
        # Area light parameters
        self.area_light_radius = 0.25
        self.area_light_radiance = [204.5762567336908, 204.5762567336908, 204.5762567336908]  #1000W/sr to 31830W/(m 2 sr)

        self._create_scene_template(only_center_mesh_instance=only_center_mesh_instance)
        calibrated_intensity = self.calibrate_light_intensity()
        self.base_light_intensity = calibrated_intensity
        print("calibrated_intensity",calibrated_intensity)
        self.radiance_clamp=15/256
        print("radiance_clamp",self.radiance_clamp)
        
        self.use_optix_denoiser = True  

        self.res = mi.ScalarVector2u(int(self.resolution), int(self.resolution))        
        self.denoiser = mi.OptixDenoiser(self.res, albedo=True, normals=True, temporal=False)


        

    def calibrate_light_intensity(self):
        if self.scene_template is None:
            self._create_scene_template()

        print("\n:")
        for key in self.params_template.keys():
            print(f"  {key}")
        
        print("\n:")
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
        
        # transform = mi.Transform4f()
        # transform = transform.translate(mi.Point3f(float(0), float(0), float(4)))
        # transform = transform.scale(self.area_light_radius)
        # self.params_template['area_light_sphere.to_world'] = transform
        if self.arealight:
            transform = mi.Transform4f()
            transform = transform.translate(mi.Point3f(float(0), float(0), float(self.lightgeight*2+2)))
            transform = transform.scale(self.area_light_radius)
            self.params_template['area_light_sphere.to_world'] = transform
        else:
            self.params_template['light.position'] = mi.Vector3f(
                float(0), float(0), float(self.lightgeight*2+2)
            )
        
        test_intensity = [5.23984641071269, 5.23984641071269, 5.23984641071269]
        if self.arealight:
            self.params_template['area_light_sphere.emitter.radiance.value'] = mi.Vector3f(*test_intensity)
        else:
            self.params_template['light.intensity.value'] = mi.Vector3f(*test_intensity)


        self.params_template.update()
        
        image = mi.render(self.scene_template)
        beauty = image[..., :3]
        
        # EXR
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
        
        print(f"correction_factor:")
        print(f"  measured_value: {measured_value:.3f}")
        print(f"  correction_factor: {correction_factor:.3f}")
        print(f"  calibrated_intensity: {calibrated_intensity}")
        
        return calibrated_intensity

    def _gaussian_blur_wrap_torch(self, height_np: np.ndarray, sigma: float) -> np.ndarray:

        if sigma <= 0:
            return height_np.astype(np.float32, copy=True)

        # kernel size  6σ 
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
        # print(dr.min(y), dr.max(y))
        return dr.clamp(y, 0.0, 1.0)
       
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
        
        x = np.linspace(-self.base_size/2, self.base_size/2, grid_size)
        y = np.linspace(-self.base_size/2, self.base_size/2, grid_size)
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

        # for i in range(grid_size):
        #     for j in range(grid_size):
        #         u = eps_u + j * (1 - 2*eps_u) / (grid_size - 1)
        #         v = eps_v + (grid_size - 1 - i) * (1 - 2*eps_v) / (grid_size - 1)
        #         temp_file.write(f"vt {u:.8f} {v:.8f}\n")        
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
        
    def _create_scene_template(self, only_center_mesh_instance = False):
        """Create scene template with mixed mirror and rotation pattern"""

        move=1-8/1024
        # Create dummy textures for initialization
        dummy_texture = mi.TensorXf(np.ones((self.resolution, self.resolution, 3), dtype=np.float32))
        
        # First define the material plane as a dict

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


        # material_plane_dict = {
        #     'type': 'obj',
        #     'filename': self.base_mesh_path,
        #     'bsdf': {
        #         'type': 'principled',
        #         'base_color': {
        #             'type': 'bitmap',
        #             'data': dummy_texture
        #         },
        #         'roughness': {
        #             'type': 'bitmap',
        #             'data': dummy_texture ,
        #             'raw': True
        #         },
        #         'metallic': {
        #             'type': 'bitmap',
        #             'data': dummy_texture ,
        #             'raw': True
        #         }
        #     }
        # }
        
        scene_dict = {
            'type': 'scene',
            'sampler': {
                'type': 'ldsampler',#'ldsampler',independent
                # 'sample_count': 64          #64#4096
            },
            
            # Create shapegroup with the material plane
            'my_shape_group': {
                'type': 'shapegroup',
                'material_plane': material_plane_dict
            },
            
            
            'sensor': {
                'type': 'perspective',
                'fov': 37,#45,
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
                    "type": "path",
                    "max_depth": 6,
                    "hide_emitters": True
                }
            }
            # "integrator": {
            #     'type': 'aov',
            #     'aovs': 'albedo:albedo, normals:sh_normal',
            #     'integrator': {
            #         'type': 'path',  
            #         'max_depth': 4,
            #         'hide_emitters': True
            #     }
            # }
        }
        

        if self.arealight:
            scene_dict['area_light_sphere'] = {
                'type': 'sphere',
                'emitter': {
                    'type': 'area',
                    'radiance': {
                        'type': 'rgb',
                        'value': self.area_light_radiance
                    }
                }
            }
        else:
            scene_dict['light'] = {
                'type': 'point',
                'position': [0.0, 0.0, 6.0],
                'intensity': {
                    'type': 'rgb',
                    'value': np.asarray(self.area_light_radiance) * (self.area_light_radius**2) * np.pi,
                }
            }


        # (x_idx, y_idx, scale_x, scale_y, rotation_degrees)
        mixed_configs = [

            (0, 0, 1, 1, 0),       
            
            (1, 0, -1, 1, 0),      #right
            (-1, 0, -1, 1, 0),     # left
            (0, 1, 1, -1, 0),      # up
            (0, -move, 1, -1, 0),     # below
            
            (1, 1, 1, 1, 180),     # right up
            (-1, 1, 1, 1, 180),   # left up
            (1, -move, 1, 1, 180),  # right down
            (-1, -move, 1, 1, 180), # left down
        ]
        
        if only_center_mesh_instance:
            mixed_configs = [mixed_configs[0]]
            

        tile_spacing = self.base_size 

        
        # for idx, (i, j, scale_x, scale_y, angle) in enumerate(mixed_configs):
        #     offset_x = i * tile_spacing
        #     offset_y = j * tile_spacing

        #     to_world = (
        #         mi.ScalarTransform4f.translate(mi.ScalarPoint3f(offset_x, offset_y, 0.0)) *
        #         mi.ScalarTransform4f.rotate([0, 0, 1], angle) *
        #         mi.ScalarTransform4f.scale(mi.ScalarVector3f(scale_x, scale_y, 1.0))
        #     )

        #     scene_dict[f'instance_mixed_{idx}'] = {
        #         'type': 'instance',
        #         'to_world': to_world,
        #         'shapegroup': { 'type': 'ref', 'id': 'my_shape_group' }
        #     }

        for idx, (i, j, scale_x, scale_y, angle) in enumerate(mixed_configs):

            transform = mi.ScalarTransform4f()
            

            transform = transform.scale([scale_x, scale_y, 1])
            

            if angle != 0:

                transform = transform.rotate([0, 0, 1], angle)
            

            # offset_x = i * tile_spacing
            # offset_y = j * tile_spacing
            # transform = transform.translate([offset_x, offset_y, 0])




            offset_x = i * tile_spacing
            offset_y = j * tile_spacing


            ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            ox, oy = offset_x, offset_y
            ox, oy =  ( ca*ox + sa*oy, -sa*ox + ca*oy )
            if scale_x != 0: ox /= scale_x
            if scale_y != 0: oy /= scale_y

            transform = mi.ScalarTransform4f()
            transform = transform.scale([scale_x, scale_y, 1])
            if angle != 0:
                transform = transform.rotate([0, 0, 1], angle)
            transform = transform.translate([ox, oy, 0])   
            
            y_top    =  +self.base_size * 0.5
            y_bottom =  -self.base_size * 0.5
            p_top    = mi.ScalarPoint3f(0.0, y_top, 0.0)
            p_bottom = mi.ScalarPoint3f(0.0, y_bottom, 0.0)
            wp_top    = transform @ p_top
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


        if False:  
            outer_configs = [

                (2, 0, 1, 1, 90),      # 
                (-2, 0, -1, 1, 90),    # 
                (0, 2, 1, 1, 0),       # 
                (0, -2, 1, -1, 180),   #

            ]
            
            for idx, (i, j, scale_x, scale_y, angle) in enumerate(outer_configs):
                transform = mi.ScalarTransform4f()
                transform = transform.scale([scale_x, scale_y, 1])
                if angle != 0:
                    transform = transform.rotate([0, 0, 1], angle)
                transform = transform.translate([i * tile_spacing, j * tile_spacing, 0])
                
                scene_dict[f'instance_outer_{idx}'] = {
                    'type': 'instance',
                    'to_world': transform,
                    'shapegroup': {
                        'type': 'ref',
                        'id': 'my_shape_group'
                    }
                }
        
        # Load scene
        self.scene_dict = scene_dict
        self.scene_template = mi.load_dict(scene_dict)
        self.params_template = mi.traverse(self.scene_template)
        
        # Store original vertices
        self.original_vertices = dr.copy(self.params_template['my_shape_group.material_plane.vertex_positions'])
        
        print(f"Scene template created with 3x3 mixed mirror/rotation pattern")
        print(f"  - Tile spacing: {tile_spacing:.3f} (overlap: {(1 - tile_spacing/self.base_size)*100:.1f}%)")
        print(f"  - Pattern: {len(mixed_configs)} instances with varied transformations")
    
    
    def _recompute_normals_seamless(self, side='bottom', pixels=3):
        """
        向量化版本：与原函数的差分与“bottom 折叠带”逻辑一致，
        只是把 Python 双重 for 循环改为 NumPy 切片/广播。
        """
        import numpy as np

        key_pos = 'my_shape_group.material_plane.vertex_positions'
        key_nrm = 'my_shape_group.material_plane.vertex_normals'

        v = np.array(self.params_template[key_pos], copy=True).astype(np.float32).reshape(-1, 3)
        G = int(round(np.sqrt(v.shape[0])))
        V = v.reshape(G, G, 3)

        p = int(pixels)
        if p < 0:
            p = 0
        if p > G-1:
            p = G-1


        Tj = np.empty_like(V, dtype=np.float32)
        Tj[:, 1:-1] = 0.5 * (V[:, 2:, :] - V[:, :-2, :])
        Tj[:, 0, :] = V[:, 1, :] - V[:, 0, :]
        Tj[:, -1, :] = V[:, -1, :] - V[:, -2, :]

        Ti = np.empty_like(V, dtype=np.float32)
        # （1..G-2）
        Ti[1:-1, :, :] = 0.5 * (V[2:, :, :] - V[:-2, :, :])
        # i==0 -> forward
        Ti[0, :, :] = V[1, :, :] - V[0, :, :]
        # i==G-1 -> backward
        Ti[-1, :, :] = V[-1, :, :] - V[-2, :, :]

        if side == 'bottom' and p > 0:
            ip  = min(p,   G-1)  
            ip1 = min(p+1, G-1)  

            # i == p 
            Ti[ip, :, :] = V[ip1, :, :] - V[ip, :, :]

            # i < p 
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

    
    # def _recompute_normals_seamless(self, side='bottom', pixels=3):
    #     import numpy as np

    #     key_pos = 'my_shape_group.material_plane.vertex_positions'
    #     key_nrm = 'my_shape_group.material_plane.vertex_normals'

    #     v = np.array(self.params_template[key_pos], copy=True).astype(np.float32).reshape(-1, 3)
    #     G = int(round(np.sqrt(v.shape[0])))
    #     V = v.reshape(G, G, 3)

    #     N = np.zeros_like(V, dtype=np.float32)
    #     p = int(pixels)

    #     def diff_x(i, j):
    #         if j == 0:
    #             return V[i, 1] - V[i, 0]
    #         elif j == G - 1:
    #             return V[i, G - 1] - V[i, G - 2]
    #         else:
    #             return 0.5 * (V[i, j + 1] - V[i, j - 1])

    #     def diff_y_bottom(i, j):
    #         if i == p:
    #             up = min(i + 1, G - 1)
    #             return V[up, j] - V[i, j]
    #         elif i < p:
    #             up0 = min(p + 1, G - 1)
    #             return V[up0, j] - V[p, j]
    #         else:
    #             if i == G - 1:
    #                 return V[i, j] - V[i - 1, j]
    #             else:
    #                 return 0.5 * (V[i + 1, j] - V[i - 1, j])

    #     for i in range(G):
    #         for j in range(G):
    #             Tj = diff_x(i, j)
    #             if side == 'bottom':
    #                 Ti = diff_y_bottom(i, j)
    #             else:
    #                 if i == 0:
    #                     Ti = V[1, j] - V[0, j]
    #                 elif i == G - 1:
    #                     Ti = V[G - 1, j] - V[G - 2, j]
    #                 else:
    #                     Ti = 0.5 * (V[i + 1, j] - V[i - 1, j])

    #             n = np.cross(Tj, Ti)
    #             ln = np.linalg.norm(n)
    #             if ln > 0:
    #                 n /= ln
    #             N[i, j] = n

    #     if side == 'bottom' and p > 0:
    #         N[:p, :, :] = N[p, :, :]

    #     if N[:, :, 2].mean() < 0:
    #         N = -N

    #     if key_nrm in self.params_template:
    #         self.params_template[key_nrm] = N.reshape(-1, 3).ravel()
    #         self.params_template.update()
    #     else:
    #         print("[warn]")

    def export_deformed_mesh(self, out_path: str):

        # import numpy as np
        # from pathlib import Path

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
            # UV
            for i in range(grid):
                for j in range(grid):
                    u = eps_u + j * (1 - 2*eps_u) / (grid - 1)
                    t = eps_v + (grid - 1 - i) * (1 - 2*eps_v) / (grid - 1)
                    f.write(f"vt {u:.8f} {t:.8f}\n")

            # face
            for i in range(grid - 1):
                for j in range(grid - 1):
                    v0 = i * grid + j + 1
                    v1 = v0 + 1
                    v2 = v0 + grid
                    v3 = v2 + 1
                    if has_vn:
                        # f v/vt/vn
                        f.write(f"f {v0}/{v0}/{v0} {v1}/{v1}/{v1} {v2}/{v2}/{v2}\n")
                        f.write(f"f {v1}/{v1}/{v1} {v3}/{v3}/{v3} {v2}/{v2}/{v2}\n")
                    else:
                        # f v/vt
                        f.write(f"f {v0}/{v0} {v1}/{v1} {v2}/{v2}\n")
                        f.write(f"f {v1}/{v1} {v3}/{v3} {v2}/{v2}\n")

        print(f"[mesh] exported: {out_path}")
        return str(out_path)



    def update_material(self, material_data, rotation_angle=0, displace=True):
        # print("1")
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
        
        if displace and ('height' in material_data or 'displacement' in material_data):
            
            if self.cached_height_map is None:
                height = material_data.get('height')
                # print("height",height)
                if isinstance(height, torch.Tensor):
                    if height.dim() == 2:
                        height = height.unsqueeze(0)
                    
                    height_np = height[0].cpu().numpy()
                    
                    if height_np.dtype == np.int16:
                        print(f"  Converting int16 height to float32")
                        height_np = height_np.astype(np.float32) / (256 * 256)
                
                if rotation_steps > 0:
                    height_np = np.rot90(height_np, rotation_steps)
                
                # print(f"  Applying Gaussian filter to heightmap (sigma={self.gaussian_sigma}) for noise reduction")
                # # height_np = self._gaussian_blur_wrap_torch(height_np, sigma=self.gaussian_sigma)
                height_np = gaussian_filter(height_np, sigma=self.gaussian_sigma ,mode='wrap')
                
                self.cached_height_map = height_np.copy() 
            else:
              
                print(f"  Using cached heightmap (rotation only)")
                height_np = np.rot90(self.cached_height_map, rotation_steps)
            # print("2")       
            self._apply_displacement(height_np)
            # print("3")  
            self.collapse_mesh_border(pixels=4, side='bottom', eps_frac=0.001, drop_z_frac=5)
            # print("4")  

            # self.collapse_mesh_border(pixels=4, side='bottom')
            # self.retile_instances_to_bbox()
        else:
            self.params_template['my_shape_group.material_plane.vertex_positions'] = dr.copy(self.original_vertices)
        
        self.params_template.update()
    def _apply_displacement(self, height_map, scale=4, reference_mode='mean'):

        import numpy as np
        from scipy.ndimage import zoom

        #  [H*W, 3]
        v0 = np.array(self.original_vertices, copy=True).reshape(-1, 3).astype(np.float32)

        grid_size = self.basemeshsize

        # height_map  (grid_size, grid_size)
        if height_map.shape != (grid_size, grid_size):
            zoom_factor = (grid_size / height_map.shape[0], grid_size / height_map.shape[1])
            height_map = zoom(height_map, zoom_factor, order=1, mode='reflect')

        H = height_map.astype(np.float32, copy=False)
        # H[:]=0
        # H[0:1024,1024]=0.2
        # H[0,  :] = 0
        # H[-1, :] = 0
        # H[:, 0]  = 0
        # H[:, -1] = 0
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

        # spans = v0.ptp(axis=0)                # [dx, dy, dz]
        axis_idx = 2#int(np.argmin(spans))      # 0->X, 1->Y, 2->Z
        disp = (H -ref_h) * float(scale)     # (H, W)
        v = v0.copy()
        v[:, axis_idx] = disp.reshape(-1)     


        # delta = self.base_size / (grid_size - 1)  

        # V0 = v0.reshape(grid_size, grid_size, 3)  
        # n_axis = np.zeros(3, dtype=np.float32)
        # n_axis[axis_idx] = 1.0


        # Tj = np.zeros_like(V0, dtype=np.float32)
        # Ti = np.zeros_like(V0, dtype=np.float32)
        # Tj[:, :-1, :] = V0[:, 1:, :] - V0[:, :-1, :]
        # Tj[:, -1, :]  = Tj[:, -2, :]
        # Ti[:-1, :, :] = V0[1:, :, :] - V0[:-1, :, :]
        # Ti[-1, :, :]  = Ti[-2, :, :]


        # dD_dj = np.zeros_like(H, dtype=np.float32)
        # dD_di = np.zeros_like(H, dtype=np.float32)
        # dD_dj[:, :-1] = disp[:, 1:] - disp[:, :-1]
        # dD_dj[:, -1]  = dD_dj[:, -2]
        # dD_di[:-1, :] = disp[1:, :] - disp[:-1, :]
        # dD_di[-1, :]  = dD_di[-2, :]

        # add_j = (dD_dj / delta)[..., None] * n_axis  # (H, W, 3)
        # add_i = (dD_di / delta)[..., None] * n_axis

        # Tj_p = Tj + add_j
        # Ti_p = Ti + add_i

        # N = np.cross(Tj_p, Ti_p)                       # (H, W, 3)
        # N /= (np.linalg.norm(N, axis=-1, keepdims=True) + 1e-8)
        # normals_flat = N.reshape(-1, 3).astype(np.float32)


        self.params_template['my_shape_group.material_plane.vertex_positions'] = v.reshape(-1)

        # key_normals = 'my_shape_group.material_plane.vertex_normals'
        # if key_normals in self.params_template:
        #     self.params_template[key_normals] = normals_flat.reshape(-1)

        self.params_template.update()
    
    # def _apply_displacement(self, height_map, scale=4, reference_mode='mean'):
    #     grid_size = int(np.sqrt(len(self.original_vertices) // 3))
    #     print("grid_size",grid_size)
        

    #     if height_map.shape != (grid_size, grid_size):

    #         # from skimage.transform import resize
    #         # height_map = resize(
    #         #     height_map,
    #         #     (grid_size, grid_size),
    #         #     order=1,                
    #         #     anti_aliasing=True,     
    #         #     preserve_range=True     
    #         # ).astype(height_map.dtype) 
    #         from scipy.ndimage import zoom
    #         zoom_factor = (grid_size / height_map.shape[0], 
    #                     grid_size / height_map.shape[1])
    #         height_map =  zoom(height_map, zoom_factor, order=1, mode='reflect')


    #     H = height_map.astype(np.float32)
    #     print("[debug] border z stats:",
    #         "top",    float(H[0, :].mean()),
    #         "prev_t", float(H[1, :].mean()),
    #         "bottom", float(H[-10, :].mean()),
    #         "prev_b", float(H[-2, :].mean()),
    #         "left",   float(H[:, 0].mean()),
    #         "prev_l", float(H[:, 1].mean()),
    #         "right",  float(H[:, -1].mean()),
    #         "prev_r", float(H[:, -2].mean()))


    #     H[0,  :] = H[1,  :]
    #     H[-1, :] = H[-2, :]
    #     H[:, 0]  = H[:, 1]
    #     H[:, -1] = H[:, -2]
        
    #     # Calculate reference height based on mode
    #     if reference_mode == 'mean':
    #         ref_height = np.mean(height_map)
    #     elif reference_mode == 'min':
    #         ref_height = np.min(height_map)
    #     elif reference_mode == 'max':
    #         ref_height = np.max(height_map)
    #     else:  # 'center'
    #         ref_height = 0.5
    #     # height_map: (H, W)，ref_height/scale 已算好


    #     H = ((height_map - ref_height) * scale).astype(np.float32).ravel()

    #     verts = np.array(self.original_vertices, copy=True)  # 3*H*W
    #     verts[2::3] = H                                     
    #     self.params_template['my_shape_group.material_plane.vertex_positions'] = verts
    #     self.params_template.update()


    #     new_vertices = dr.copy(self.original_vertices)
    #     for i in range(grid_size):
    #         for j in range(grid_size):
    #             # print(i,j)
    #             idx = (i * grid_size + j) * 3 + 2
    #             # Offset from reference plane
    #             new_vertices[idx] = (height_map[i, j] - ref_height) * scale
        
    #     self.params_template['my_shape_group.material_plane.vertex_positions'] = new_vertices
    #     self.params_template.update()
        # H = height_map.astype(np.float32)
        # delta = self.base_size / (grid_size - 1)
        # s = float(scale)

        # # dHx = (np.roll(H, -1, axis=1) - np.roll(H,  1, axis=1)) * 0.5
        # # dHy = (np.roll(H, -1, axis=0) - np.roll(H,  1, axis=0)) * 0.5

        # dHx = np.empty_like(H, dtype=np.float32)
        # dHy = np.empty_like(H, dtype=np.float32)

        # dHx[:, 1:-1] = 0.5 * (H[:, 2:] - H[:, :-2])
        # dHy[1:-1, :] = 0.5 * (H[2:, :] - H[:-2, :])
        # dHx[:,  0]  = 0.0
        # dHx[:, -1]  = 0.0
        # dHy[ 0, :]  = 0.0
        # dHy[-1, :]  = 0.0


        # # dHx[:, 0]  = H[:, 1]   - H[:, 0]  
        # # dHx[:, -1] = H[:, -1]  - H[:, -2]   
        # # dHy[0, :]  = H[1, :]   - H[0, :]   
        # # dHy[-1, :] = H[-1, :]  - H[-2, :]    
        # nx = -(s / delta) * dHx
        # ny = -(s / delta) * dHy
        # nz = np.ones_like(H, dtype=np.float32)

        # N = np.stack([nx, ny, nz], axis=-1)
        # N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-8
        # normals_flat = N.reshape(-1, 3).astype(np.float32)

        # self.params_template['my_shape_group.material_plane.vertex_normals'] = mi.TensorXf(normals_flat.ravel())
        # self.params_template.update()
    
    # def render_btf_sequence(self, output_path, total_frames=129, save_frames=False, rotation_angle=0, suffix="", save_exr=False):
    #     output_path = Path(output_path)
    #     output_path.mkdir(parents=True, exist_ok=True)
        
    #     phase1_frames = total_frames // 2
    #     phase3_frames = total_frames - phase1_frames
        
    #     frames = []
    #     frame_idx = 0
        
    #     camera_pos = [0.0, 0.0, self.camera_height]
    #     light_horizontal_radius = self.light_radius * np.sin(np.pi / 3)
    #     base_distance = self.lightgeight*2
        
    #     # Phase 1: Light rotation
    #     for i in range(phase1_frames):
    #         t = i / (phase1_frames - 1) if phase1_frames > 1 else 0
    #         theta = 2 * np.pi * t

    #         light_pos = [
    #             self.lightgeight*light_horizontal_radius * np.cos(theta),
    #             self.lightgeight*light_horizontal_radius * np.sin(theta),
    #             self.lightgeight*2
    #         ]
    #         actual_distance = np.sqrt(light_pos[0]**2 + light_pos[1]**2 + light_pos[2]**2)
    #         intensity_correction = (actual_distance/base_distance) ** 2
    #         corrected_intensity = [float(val * intensity_correction) for val in self.base_light_intensity]
    #         self.params_template['sensor.camera_position'] = mi.Vector3f(
    #             float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])
    #         )
    #         if self.arealight:
    #             transform = mi.Transform4f()
    #             transform = transform.translate(mi.Point3f(float(light_pos[0]), float(light_pos[1]), float(light_pos[2])))

    #             transform = transform.scale(self.area_light_radius)
    #             self.params_template['area_light_sphere.to_world'] = transform

    #             self.params_template['area_light_sphere.emitter.radiance.value'] = mi.Vector3f(*corrected_intensity)
    #         else:
    #             self.params_template['light.position'] = mi.Vector3f(
    #                 float(light_pos[0]), float(light_pos[1]), float(light_pos[2])
    #             )
    #             self.params_template['light.intensity.value'] = mi.Vector3f(*corrected_intensity)



    #         self.params_template.update()


            
    #         seed = np.random.randint(0, 2**31 - 1)  #
    #         # image = mi.render(self.scene_template, seed=seed)
    #         image_all = mi.render(self.scene_template, seed=seed)
    #         beauty  = image_all[..., 0:3]
    #         albedo  = image_all[..., 3:6]
    #         normals = image_all[..., 6:9]
                        
    #         if self.use_optix_denoiser:
    #             image = self.denoiser(beauty, albedo=albedo, normals=normals)
    #         else:
    #             image = beauty
                    
            
    #         frames.append(image)
            
    #         if save_frames:
    #             frame_path = output_path / f"btf_frame_{suffix}_{frame_idx:04d}.png"
    #             mi.util.write_bitmap(str(frame_path), image)
            
    #         if save_exr:
    #             exr_path = output_path / f"btf_frame_{suffix}_{frame_idx:04d}.exr"
    #             mi.util.write_bitmap(str(exr_path), image)
            
    #         frame_idx += 1
        
    #     # Phase 2: Static light, camera rotation
    #     light_pos = [0.0, 0.0, self.lightgeight*2]
        
    #     for i in range(phase3_frames):
    #         t = i / (phase3_frames - 1) if phase3_frames > 1 else 0
    #         theta = 2 * np.pi * t
    #         camera_pos = [
    #             self.camera_radius * np.cos(theta),
    #             -self.camera_radius * np.sin(theta),
    #             self.camera_height
    #         ]
            
    #         self.params_template['sensor.camera_position'] = mi.Vector3f(
    #             float(camera_pos[0]), float(camera_pos[1]), float(camera_pos[2])
    #         )
            
    #         # Keep light at center position
    #         if self.arealight:
    #             transform = mi.Transform4f()
    #             transform = transform.translate(mi.Point3f(float(light_pos[0]), float(light_pos[1]), float(light_pos[2])))
    #             transform = transform.scale(self.area_light_radius)
    #             self.params_template['area_light_sphere.to_world'] = transform
    #             self.params_template['area_light_sphere.emitter.radiance.value'] = mi.Vector3f(*self.base_light_intensity)
    #         else:
    #             self.params_template['light.position'] = mi.Vector3f(
    #                 float(light_pos[0]), float(light_pos[1]), float(light_pos[2])
    #             )
    #             self.params_template['light.intensity.value'] = mi.Vector3f(*self.base_light_intensity)






 
    #         self.params_template.update()

    #         seed = np.random.randint(0, 2**31 - 1)  #
    #         # image = mi.render(self.scene_template, seed=seed)
    #         image_all = mi.render(self.scene_template, seed=seed)
    #         beauty  = image_all[..., 0:3]
    #         albedo  = image_all[..., 3:6]
    #         normals = image_all[..., 6:9]
                        
    #         if self.use_optix_denoiser:
    #             image = self.denoiser(beauty, albedo=albedo, normals=normals)
    #         else:
    #             image = beauty
                    
            
    #         frames.append(image)
            
    #         if save_frames:
    #             frame_path = output_path / f"btf_frame_{suffix}_{frame_idx:04d}.png"
    #             mi.util.write_bitmap(str(frame_path), image)
            
    #         if save_exr:
    #             exr_path = output_path / f"btf_frame_{suffix}_{frame_idx:04d}.exr"
    #             mi.util.write_bitmap(str(exr_path), image)
            
    #         frame_idx += 1
        
    #     video_path = output_path / f'btf_sequence{suffix}.mp4'
    #     self._create_video_from_frames(frames, video_path)
        
    #     return video_path

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
                
        np.save(output_path, state)


    def render_and_save_arbitrary_maps(self, arbitrary_maps, output_path, suffix, frame_idx):
        [os.makedirs(output_path / 'maps'/k , exist_ok=True) for k in arbitrary_maps]
        w,h = self.resolution, self.resolution
        arb_integrator = ArbitraryIntegrator(mi.Properties())
        mi.render(self.scene_template, integrator=arb_integrator, seed=0, spp=1)
        vals = arb_integrator.get_arb_vals(w,h)     
        for k in arbitrary_maps:
            img = ensure_RGB_dims(vals[k], let_1_be_1=True)
            mi.Bitmap(img).write(str(output_path / 'maps'/ k / f"btf_frame_{suffix}_{frame_idx:04d}.exr"))
        del vals
        del arb_integrator
        gc.collect()


    def render_btf_sequence(self, output_path, total_frames=129, save_frames=False, rotation_angle=0, suffix="", save_exr=False, val_states=None, arbitrary_maps=None, spp = 128):
        render_path_traced = not arbitrary_maps
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
        if render_path_traced:
            try:
                ff_proc, encoder_name = self._open_ffmpeg(video_path, w, h, fps=30, prefer_nvenc=False, cq=23)
            except Exception as e:
                print(f"[FFmpeg] fail back to libx264 ...")
                ff_proc, encoder_name = self._open_ffmpeg(video_path, w, h, fps=30, prefer_nvenc=False, cq=23)

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

            if val_states is not None:
                load_scene_state(val_states[frame_idx], self.params_template, not self.arealight)


            if arbitrary_maps:
                self.render_and_save_arbitrary_maps(arbitrary_maps, output_path, suffix, frame_idx)


            if render_path_traced:
                seed = np.random.randint(0, 2**31 - 1)
                image_all = mi.render(self.scene_template, seed=seed, spp=spp)
                # image_all = mi.render(self.scene_template)

                beauty  = image_all[..., 0:3]
                albedo  = image_all[..., 3:6]
                #TODO: debug the spehere in albedo
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
                    ff_proc, encoder_name = self._open_ffmpeg(video_path, w, h, fps=30, prefer_nvenc=False, cq=23)
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
            if val_states is not None:
                load_scene_state(val_states[frame_idx], self.params_template, not self.arealight)

            if arbitrary_maps:
                self.render_and_save_arbitrary_maps(arbitrary_maps, output_path, suffix, frame_idx)
            
            if render_path_traced:

                seed = np.random.randint(0, 2**31 - 1)
                image_all = mi.render(self.scene_template, seed=seed, spp=spp)
                beauty  = image_all[..., 0:3]
                albedo  = image_all[..., 3:6]
                normals = image_all[..., 6:9]



                image = self.denoiser(beauty, albedo=albedo, normals=normals) if self.use_optix_denoiser else beauty

                srgb = self._linear_to_srgb_gpu(image)
                srgb_np = np.array(srgb)
                rgb_uint8 = (srgb_np * 255.0 + 0.5).astype(np.uint8)
                ff_proc.stdin.write(rgb_uint8.tobytes())
    
                if save_frames:
                    frame_path = output_path / "png" / f"btf_frame_{suffix}_{frame_idx:04d}.png"
                    mi.util.write_bitmap(str(frame_path), image)

                if save_exr:
                    exr_path = output_path / 'exr' / f"btf_frame_{suffix}_{frame_idx:04d}.exr"
                    mi.util.write_bitmap(str(exr_path), image)

                if save_frames or save_exr:
                    self.save_scene_state(output_path / 'scene_state' /f"btf_frame_{suffix}_{frame_idx:04d}.npy")
            
            frame_idx += 1

        if render_path_traced:
            try:
                ff_proc.stdin.close()
            finally:
                ff_proc.wait()

        print(f"Video saved: {video_path}")
        return video_path


    def linear_to_srgb(self,x):
            x = np.clip(x, 0.0, 1.0)
            a = 0.055
            return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * np.power(x, 1/2.4) - a)
            
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
        
    def process_local_dataset(self, parquet_dir, output_base_dir, max_materials=None, start_idx=0, save_exr=False, save_frames=False, process_mode='all', n_frames=129, material_name = None, rotations=[0,90,180,270], val_states=None, arbitrary_maps=None, spp = 128):
        output_base = Path(output_base_dir)
        output_base.mkdir(parents=True, exist_ok=True)
        # parquet_dir = Path(parquet_dir)

        # print(f"Loading dataset from: {parquet_dir}")
        # ds = load_dataset(
        #     "parquet",
        #     data_dir=str(parquet_dir),
        #     split="train",
        #     streaming=True
        # )

        ds = load_dataset(
            "gvecchio/MatSynth", 
            split='train',
        )
        assert material_name is not None

        for i, name in enumerate(ds['metadata']['name']):
            if name == material_name:
                break
        
        ds = [ds[i]]

        material_count = 0
        current_idx = 0

        for material_data in ds:
            # print('current_idx', current_idx, 'start_idx', start_idx)
            # if current_idx < start_idx:
            #     current_idx += 1
            #     continue

            # should_process = False
        
            # if process_mode == 0:
            #     should_process = True
            # elif process_mode == 1:
            #     should_process = current_idx % 2 == 1
            # elif process_mode == 2:
            #     should_process = current_idx % 2 == 0
            # print('current_idx', should_process)
            # print(max_materials, material_count)

            # if not should_process:
            #     current_idx += 1
            #     continue

            # if max_materials and material_count >= max_materials:
            #     break
            material_name = material_data['metadata'].get('name', f'material_{current_idx:04d}')
            height_factor = material_data['metadata'].get('height_factor')
            height_mean = material_data['metadata'].get('height_mean')
            print(f"\nProcessing [{current_idx + 1}]: {material_name}")

            output_path = output_base / material_name
            all_exist = True
            for angle in rotations:

                suffix = f"_{angle}deg" if angle > 0 else ""
                if not (output_path / f'btf_sequence{suffix}.mp4').exists():
                    all_exist = False
                    break
            
            if all_exist:
                print(f"  Skipping - already processed (all rotations)")
                current_idx += 1
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
                            texture_np = np.array(texture)
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
                        texture=(texture-height_mean)*height_factor

                    if texture.shape[-2:] != (self.resolution, self.resolution) and key != 'height':
                        if texture.dim() == 2:
                            texture = texture.unsqueeze(0)
                        texture = TF.resize(texture, (self.resolution, self.resolution))
                    
                    processed_data[key] = texture
                    
            self.cached_height_map = None
            for i, angle in enumerate(rotations):
                # self.debug_vertices()
                self.update_material(processed_data, rotation_angle=angle)
                # self.debug_vertices()
                # self.export_deformed_mesh(output_path / f"mesh{suffix}.obj")
                
                suffix = f"_{angle}deg" if angle > 0 else ""
                print(f"  Rendering rotation {angle} degrees...")
                video_path = self.render_btf_sequence(output_path, rotation_angle=angle, suffix=suffix, save_exr=save_exr, save_frames=save_frames, total_frames=n_frames, val_states=val_states, arbitrary_maps=None, spp=spp)

                self.update_material(processed_data, rotation_angle=angle, displace=False)
                video_path_maps = self.render_btf_sequence(output_path, rotation_angle=angle, suffix=suffix, save_exr=save_exr, save_frames=save_frames, total_frames=n_frames, val_states=val_states, arbitrary_maps=arbitrary_maps, spp=spp)

                print(f"  Completed: {video_path}, maps at {video_path_maps}")

            material_count += 1
            current_idx += 1

            del processed_data
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"\nProcessed {material_count} materials")


    def _compute_normals_rows_subset(self, V, i0, i1):
        import numpy as np
        Gy, Gx, _ = V.shape
        i0 = max(0, int(i0)); i1 = min(Gy - 1, int(i1))
        cnt = i1 - i0 + 1

        V_row = V[i0:i1 + 1]                           # (cnt, Gx, 3)

        Tj = np.empty_like(V_row, dtype=np.float32)
        Tj[:, 1:-1] = 0.5 * (V_row[:, 2:] - V_row[:, :-2])
        Tj[:, 0]    = V_row[:, 1] - V_row[:, 0]
        Tj[:, -1]   = V_row[:, -1] - V_row[:, -2]

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
        j0 = max(0, int(j0)); j1 = min(Gx - 1, int(j1))
        cnt = j1 - j0 + 1
        js = np.arange(j0, j1 + 1)

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
            Ti[0, k]        = V[1, j] - V[0, j]
            Ti[Gy - 1, k]   = V[Gy - 1, j] - V[Gy - 2, j]

        N = np.cross(Tj, Ti)
        L = np.linalg.norm(N, axis=-1, keepdims=True)
        N = N / np.maximum(L, 1e-8)
        return N.astype(np.float32)


    # def collapse_mesh_border(self, pixels=3, side='bottom',
    #                         eps_frac=0.1, drop_z_frac=0.5):

    #     import numpy as np
    #     key_pos = 'my_shape_group.material_plane.vertex_positions'
    #     key_nrm = 'my_shape_group.material_plane.vertex_normals'


    #     v = np.array(self.params_template[key_pos], copy=True).astype(np.float32)
    #     n_verts = v.size // 3
    #     G = int(round(n_verts ** 0.5))
    #     V = v.reshape(G, G, 3)


    #     px = self.base_size / max(G - 1, 1)
    #     band = float(eps_frac) * px
    #     drop = float(drop_z_frac) * px
    #     p = int(pixels)

    #     if side == 'bottom':
    #         y_ref = V[p, :, 1].copy()
    #         for k in range(p):
    #             V[k, :, 1] = y_ref - band * (p - k)
    #             V[k, :, 2] -=      drop * (p - k)
    #         i0, i1 = 0, min(p, G - 1)
    #         extra_row = (p + 1) if (p + 1) < G else None

    #     elif side == 'top':
    #         y_ref = V[-p-1, :, 1].copy()
    #         for k in range(p):
    #             V[-1-k, :, 1] = y_ref + band * (p - k)
    #             V[-1-k, :, 2] -=      drop * (p - k)
    #         i0 = max(G - p - 1, 0)     
    #         i1 = G - 1
    #         extra_row = (i0 - 1) if (i0 - 1) >= 0 else None

    #     elif side == 'left':
    #         x_ref = V[:, p, 0].copy()
    #         for k in range(p):
    #             V[:, k, 0] = x_ref - band * (p - k)
    #             V[:, k, 2] -=      drop * (p - k)
    #         j0, j1 = 0, min(p, G - 1)
    #         extra_col = (p + 1) if (p + 1) < G else None

    #     elif side == 'right':
    #         x_ref = V[:, -p-1, 0].copy()
    #         for k in range(p):
    #             V[:, -1-k, 0] = x_ref + band * (p - k)
    #             V[:, -1-k, 2] -=      drop * (p - k)
    #         j0 = max(G - p - 1, 0)     
    #         j1 = G - 1
    #         extra_col = (j0 - 1) if (j0 - 1) >= 0 else None

    #     else:
    #         raise ValueError("side must be 'bottom'/'top'/'left'/'right'")

    #     if key_nrm in self.params_template:
    #         N = np.array(self.params_template[key_nrm], copy=True).astype(np.float32)\
    #             .reshape(G, G, 3)

    #         if side in ('bottom', 'top'):
  
    #             N[i0:i1 + 1] = self._compute_normals_rows_subset(V, i0, i1)
    #             if extra_row is not None:
    #                 N[extra_row:extra_row + 1] = self._compute_normals_rows_subset(V, extra_row, extra_row)
    #             if side == 'bottom' and p > 0:
    #                 N[:p] = N[p]
    #             if side == 'top' and p > 0:
    #                 N[G - p:G] = N[G - p - 1]

    #         else:  # left/right
    #             N[:, j0:j1 + 1] = self._compute_normals_cols_subset(V, j0, j1)
    #             if extra_col is not None:
    #                 N[:, extra_col:extra_col + 1] = self._compute_normals_cols_subset(V, extra_col, extra_col)
    #             if side == 'left' and p > 0:
    #                 N[:, :p] = N[:, p:p + 1]
    #             if side == 'right' and p > 0:
    #                 N[:, G - p:G] = N[:, G - p - 1:G - p]


    #         self.params_template[key_nrm] = N.reshape(-1)


    #     self.params_template[key_pos] = V.reshape(-1)
    #     self.params_template.update()

    #     self._collapsed = {'side': side, 'pixels': p, 'band_thick': float(band)}


    def collapse_mesh_border(self, pixels=3, side='bottom',
                            eps_frac=0.1,     #  eps_frac * pixel_world 
                            drop_z_frac=0.5   # Z  drop_z_frac * pixel_world * k
                            ):


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
            y_ref = V[-pixels-1, :, 1].copy()
            for k in range(pixels):  
                y_new = y_ref + band_thick * (pixels - k)
                V[-1-k, :, 1] = y_new
                V[-1-k, :, 2] -= drop_z_unit * (pixels - k)
        elif side == 'left':
            x_ref = V[:, pixels, 0].copy()
            for k in range(pixels):
                x_new = x_ref - band_thick * (pixels - k)
                V[:, k, 0] = x_new
                V[:, k, 2] -= drop_z_unit * (pixels - k)
        elif side == 'right':
            x_ref = V[:, -pixels-1, 0].copy()
            for k in range(pixels):
                x_new = x_ref + band_thick * (pixels - k)
                V[:, -1-k, 0] = x_new
                V[:, -1-k, 2] -= drop_z_unit * (pixels - k)
        else:
            raise ValueError("side must be bottom/top/left/right")

        self.params_template[key] = V.reshape(-1)
        self.params_template.update()


        self._collapsed = {'side': side, 'pixels': int(pixels), 'band_thick': float(band_thick)}
        # print("compute normal")
        self._recompute_normals_seamless(side=side, pixels=pixels)
        # print("compute normal2")

    def retile_instances_to_bbox(self, ignore_collapsed_band=True):

        import numpy as np, math, mitsuba as mi

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
                    y_max_vis = float(V[-p-1, :, 1].max())
                    span_y = y_max_vis - y_min_vis
                elif side == 'left':
                    x_min_vis = float(V[:, p, 0].min())
                    x_max_vis = float(V[:, -1, 0].max())
                    span_x = x_max_vis - x_min_vis
                elif side == 'right':
                    x_min_vis = float(V[:, 0, 0].min())
                    x_max_vis = float(V[:, -p-1, 0].max())
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
            ox, oy = ( ca*offset_x + sa*offset_y, -sa*offset_x + ca*offset_y )
            if scale_x != 0: ox /= scale_x
            if scale_y != 0: oy /= scale_y

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

        import subprocess, shutil
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

    def _open_ffmpeg(self, path, w, h, fps=30, prefer_nvenc=True, cq=23):

        import subprocess, shutil
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
        """Write one RGB uint8 frame (HxWx3) into ffmpeg stdin."""
        proc.stdin.write(rgb_uint8.tobytes())

    def _close_ffmpeg(self, proc):
        """Finalize ffmpeg process."""
        proc.stdin.close()
        proc.wait()


def process_materials_worker(gpu_id, indices, parquet_dir, output_dir, resolution, total_gpus, save_exr=False):
    print(f"\nWorker on GPU {gpu_id} starting, processing {len(indices)} materials")
    
    renderer = StreamingBTFRenderer(resolution=resolution, gpu_id=gpu_id)
    
    renderer.process_local_dataset(
        parquet_dir=parquet_dir,
        output_base_dir=output_dir,
        max_materials=len(indices),
        start_idx=indices[0],
        save_exr=save_exr
    )
    
    print(f"\nWorker on GPU {gpu_id} completed")


def main():
    import sys
    
    # Check if local parquet directory is provided
    with dr.suspend_grad():
        if True:
            # Use local parquet files
            parquet_dir =  "/mnt/iusers01/fatpou01/compsci01/v67771bx/scratch/MatSynth/data"
            output_dir = sys.argv[2] if len(sys.argv) > 2 else "/mnt/iusers01/fatpou01/compsci01/v67771bx/scratch/btf_dataset_output6"
            max_materials = int(sys.argv[3]) if len(sys.argv) > 3 else 5700
            start_idx = int(sys.argv[4]) if len(sys.argv) > 4 else 0
            num_gpus = int(sys.argv[5]) if len(sys.argv) > 5 else 1
            save_exr = sys.argv[6].lower() == 'true' if len(sys.argv) > 6 else True  
            process_mode=sys.argv[7] if len(sys.argv) > 6 else 0
            
            print(f"Processing local dataset from: {parquet_dir}")
            print(f"Output directory: {output_dir}")
            print(f"Max materials: {max_materials if max_materials else 'All'}")
            print(f"Starting from index: {start_idx}")
            print(f"Using {num_gpus} GPUs")
            print(f"Save EXR frames: {save_exr}")
            if save_exr:
                print(f"  WARNING: Saving all frames as EXR will use significant disk space!")
                print(f"  Estimated: ~{1024 * 1024 * 3 * 4 * 129 / (1024**2):.1f} MB per material rotation")
            
            if num_gpus > 1 and torch.cuda.device_count() >= num_gpus:
                print(f"Multi-GPU mode: Using {num_gpus} GPUs")
                
                if max_materials:
                    materials_per_gpu = max_materials // num_gpus
                    remainder = max_materials % num_gpus
                else:
                    print("Warning: Multi-GPU mode without max_materials may not balance load properly")
                    materials_per_gpu = 1000
                    remainder = 0
                
                processes = []
                for gpu_id in range(num_gpus):
                    gpu_start_idx = start_idx + gpu_id * materials_per_gpu
                    gpu_materials = materials_per_gpu + (1 if gpu_id < remainder else 0)
                    
                    if max_materials:
                        indices = list(range(gpu_start_idx, min(gpu_start_idx + gpu_materials, start_idx + max_materials)))
                    else:
                        indices = list(range(gpu_start_idx, gpu_start_idx + materials_per_gpu))
                    p = mp.Process(
                        target=process_materials_worker,
                        args=(gpu_id, indices, parquet_dir, output_dir, 1024, num_gpus, save_exr)
                    )
                    p.start()
                    processes.append(p)
                
                for p in processes:
                    p.join()
                
                print("\nAll GPU workers completed")
                if save_exr:
                    print("Note: All frames have been saved as EXR files")
            else:
                print(f"Single GPU mode")
                print(f"OptiX denoiser: Enabled")  
                renderer = StreamingBTFRenderer(resolution=1024)
                renderer.use_optix_denoiser = True  
                renderer.process_local_dataset(
                    parquet_dir=parquet_dir,
                    output_base_dir=output_dir,
                    max_materials=max_materials,
                    start_idx=start_idx,
                    save_exr=save_exr,
                    process_mode=process_mode
                )
                if save_exr:
                    print("Note: All frames have been saved as EXR files")

if __name__ == "__main__":
    main()