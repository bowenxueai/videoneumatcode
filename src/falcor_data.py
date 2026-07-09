import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from torchvision.transforms.functional import gaussian_blur


# TrainingRecord parsing - numpy required for struct parsing from Falcor buffer

TRAINING_RECORD_DTYPE = np.dtype([
    ('frame_id', np.int32),
    ('valid', np.float32),
    ('material_id', np.int32),
    ('_pad0', np.int32),
    ('uv', np.float32, 2),
    ('wi', np.float32, 3),
    ('wo', np.float32, 3),
    ('mip_level', np.int32),
    ('target', np.float32, 3),
    ('uv_offset', np.float32, 2),
])


@dataclass
class TrainingBatch:
    """Parsed training batch with named fields for ML training."""
    material_name: str
    batch_index: int
    batch_shape: Tuple[int, int]
    valid: np.ndarray
    uv: np.ndarray
    wi: np.ndarray
    wo: np.ndarray
    target: np.ndarray
    material_id: np.ndarray
    mip_level: np.ndarray
    uv_offset: np.ndarray


def parse_training_record(raw_data: np.ndarray, material_name: str, batch_index: int) -> TrainingBatch:
    """Parse raw training record data into structured TrainingBatch."""
    batch_shape = raw_data.shape[:2]
    flat = raw_data.view(np.uint8).reshape(-1, 72)
    records = np.frombuffer(flat.tobytes(), dtype=TRAINING_RECORD_DTYPE).reshape(batch_shape)
    return TrainingBatch(
        material_name=material_name,
        batch_index=batch_index,
        batch_shape=batch_shape,
        valid=records['valid'] > 0.5,
        uv=records['uv'],
        wi=records['wi'],
        wo=records['wo'],
        target=records['target'],
        material_id=records['material_id'],
        mip_level=records['mip_level'],
        uv_offset=records['uv_offset'],
    )


# Falcor Data Generator (per-process singleton, safe for multi-GPU)

_FALCOR_GENERATOR = None
_FALCOR_PROCESS_INDEX = None


def _tensor_to_spy(spy, device, tensor: torch.Tensor, add_channel_dim: bool = False):
    """Convert CUDA torch tensor to spy.Tensor via GPU-to-GPU copy."""
    tensor = tensor.detach().contiguous().float()
    if add_channel_dim and tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    
    shape = tuple(tensor.shape)
    usage = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access | spy.BufferUsage.shared
    spy_tensor = spy.Tensor.zeros(device, dtype='float', shape=shape, usage=usage)
    device.sync_to_device()
    spy_tensor.copy_from_torch(tensor)
    return spy_tensor


def get_falcor_generator(
    res: int = 256, 
    batch_shape: Tuple[int, int] = (32, 32), 
    spp: int = 16, 
    log_level: str = 'error',
    process_index: int = 0,
):
    """Get or create the Falcor data generator (per-process singleton).
    
    With accelerate multi-GPU, each process has its own singleton on its assigned GPU.
    Uses create_torch_device to properly share CUDA context with PyTorch.
    """
    global _FALCOR_GENERATOR, _FALCOR_PROCESS_INDEX
    
    import os
    
    if _FALCOR_GENERATOR is not None and _FALCOR_PROCESS_INDEX == process_index:
        return _FALCOR_GENERATOR
    
    _FALCOR_PROCESS_INDEX = process_index
    print(f"[FalcorGenerator] Initializing for process {process_index} on GPU {process_index} (PID: {os.getpid()})")
    
    import slangpy as spy
    from slangpy import create_torch_device, DeviceType
    from falcor2 import ImporterMesh
    from falcor2.minitracer.tools import create_scene, Renderer
    
    log_levels = {
        'none': spy.LogLevel.none,
        'fatal': spy.LogLevel.fatal,
        'error': spy.LogLevel.error,
        'warn': spy.LogLevel.warn,
        'info': spy.LogLevel.info,
        'debug': spy.LogLevel.debug,
    }
    spy.Logger.get().level = log_levels.get(log_level, spy.LogLevel.error)
    
    neuralappearance_path = Path(__file__).parent.parent.parent / 'neuralappearance'
    falcor2_path = neuralappearance_path / 'external' / 'falcor2'
    
    if not falcor2_path.exists():
        raise RuntimeError(f"falcor2 not found at {falcor2_path}")
    
    slangpy_shader_path = str(falcor2_path / 'external' / 'slangpy' / 'slangpy' / 'slang')
    include_paths = [
        slangpy_shader_path,
        str(falcor2_path / 'slang'),
        str(neuralappearance_path / 'neuralappearance'),
        str(neuralappearance_path / 'neuralappearance' / 'datagen'),
    ]
    
    device = create_torch_device(
        type=DeviceType.cuda,
        torch_device=process_index,
        include_paths=include_paths,
        enable_print=False,
        enable_hot_reload=False,
    )
    
    print(f"[FalcorGenerator] Created device on {device.info.adapter_name} for process {process_index}")
    
    class ExtendedRenderer(Renderer):
        def __init__(self, device):
            super().__init__(device)
            self.module = self.load_module('extended_path_tracer')
            self.render_training_record_fn = self.find_render_function(self.module, 'render_training_record')
    
    class FalcorGenerator:
        def __init__(self):
            self.device = device
            self.res = res
            self.batch_shape = batch_shape
            self.spp = spp
            self.quad_side = 4.0
            self.process_index = process_index
            self.renderer = ExtendedRenderer(device)
            self.renderer.max_depth = 3
            self.renderer.tone_map = False
            self.renderer.enable_mis = False
            self.renderer.enable_nee = True
            self.renderer.enable_env_map = False
            self.renderer.background_color = spy.float3(0.0, 0.0, 0.0)
            self.training_buffer = spy.NDBuffer(
                device, dtype=self.renderer.module['TrainingRecord'], shape=batch_shape
            )
            self.current_scene = None
            self._current_material_key = None
            self.enable_cache = True  # Set to False to disable material caching
            self._spy = spy
            self._create_scene = create_scene
            self._ImporterMesh = ImporterMesh
            self._cached_indices = None  # Pre-computed triangle indices
        
        def load_material(
            self, 
            basecolor: torch.Tensor, 
            roughness: torch.Tensor, 
            metallic: torch.Tensor, 
            height: torch.Tensor, 
            normal: Optional[torch.Tensor] = None, 
            height_mean: float = 0.5, 
            height_factor: float = 10.0/255,
            material_name: str = None,
        ):
            """Load material textures into Falcor. Expects CUDA tensors."""
            spy = self._spy
            
            # Check cache (can disable with falcor_gen.enable_cache = False)
            if self.enable_cache and material_name is not None and material_name == self._current_material_key and self.current_scene is not None:
                return
            
            # Height processing on GPU
            height_t = height.float()
            if height_t.ndim == 3:
                height_t = height_t[..., 0]
            
            height_t = (height_t - height_mean) * height_factor * self.quad_side
            
            sigma = 0.5 * height_t.shape[0] / self.res
            kernel_size = int(6 * sigma) | 1
            kernel_size = max(kernel_size, 3)
            height_t = height_t.unsqueeze(0).unsqueeze(0)
            height_t = gaussian_blur(height_t, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
            height_t = height_t.squeeze(0).squeeze(0)
            height_t = height_t - height_t.mean()
            
            # Mesh creation requires numpy (ImporterMesh API)
            processed_height = height_t.cpu().numpy().astype(np.float32)
            
            scene = self._create_scene(self.device)
            quad = self._create_quad(scene, processed_height)
            
            material = scene.create_material('default')
            material.specular = scene.white_texture
            
            # Textures via GPU-to-GPU copy
            material.albedo = _tensor_to_spy(spy, self.device, basecolor, add_channel_dim=False)
            
            r = roughness.squeeze(-1) if roughness.ndim == 3 else roughness
            material.roughness = _tensor_to_spy(spy, self.device, r, add_channel_dim=True)
            
            m = metallic.squeeze(-1) if metallic.ndim == 3 else metallic
            material.metallic = _tensor_to_spy(spy, self.device, m, add_channel_dim=True)
            
            if normal is not None:
                material.normal = _tensor_to_spy(spy, self.device, normal, add_channel_dim=False)
            
            quad.material = material
            self.current_scene = scene
            self._current_material_key = material_name
        
        def _create_quad(self, scene, height_map):
            """Create quad mesh. Uses numpy for ImporterMesh API."""
            half_side = self.quad_side / 2.0
            u = np.linspace(-half_side, half_side, self.res, dtype=np.float32)
            v = np.linspace(-half_side, half_side, self.res, dtype=np.float32)
            uu, vv = np.meshgrid(u, v)
            zz = height_map if height_map is not None else np.zeros_like(uu)
            positions = np.stack([uu, vv, zz], axis=-1)
            
            uv_u = np.linspace(0, 1, self.res, dtype=np.float32)
            uv_v = np.linspace(0, 1, self.res, dtype=np.float32)
            uv_uu, uv_vv = np.meshgrid(uv_u, uv_v, indexing='xy')
            uv_coords = np.stack([uv_uu, uv_vv], axis=-1)
            
            # Cache indices - they don't change between materials
            if self._cached_indices is None:
                # Vectorized index generation (much faster than Python loop)
                i = np.arange(self.res - 1)
                j = np.arange(self.res - 1)
                ii, jj = np.meshgrid(i, j, indexing='ij')
                k = ii * self.res + jj
                k_right = k + 1
                k_up = k + self.res
                k_up_right = k + self.res + 1
                tri1 = np.stack([k, k_right, k_up], axis=-1)
                tri2 = np.stack([k_right, k_up_right, k_up], axis=-1)
                self._cached_indices = np.concatenate([tri1.reshape(-1, 3), tri2.reshape(-1, 3)], axis=0).astype(np.uint32)
            
            im = self._ImporterMesh.create(
                subgeometries=[self._cached_indices],
                attributes={'position': positions.reshape(-1, 3), 'tex_coord_0': uv_coords.reshape(-1, 2)}
            )
            im.add_normals_from_faces()
            im.add_tangents_from_normals()
            return scene.import_mesh_instance(im)
        
        def generate_batch(self, batch_index: int) -> np.ndarray:
            """Generate a batch of BRDF training samples."""
            if self.current_scene is None:
                raise ValueError("No material loaded")
            
            self.renderer.call(
                self.renderer.render_training_record_fn,
                scene=self.current_scene,
                material_id=0,
                batch_pos=self._spy.grid(self.batch_shape),
                batch_res=self.batch_shape[0],
                side=self.quad_side,
                record=self.training_buffer,
                spp=self.spp,
                batch_index=batch_index,
                latent_res=self.res,
                mip_level=0,
            )
            
            self.device.sync_to_device()
            return self.training_buffer.to_numpy()
        
        def unload_material(self):
            """Unload current material to free resources."""
            self.current_scene = None
            self._current_material_key = None
    
    _FALCOR_GENERATOR = FalcorGenerator()
    return _FALCOR_GENERATOR
