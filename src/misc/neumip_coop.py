"""
NeuMIP network implementation using DrJit cooperative vectors (inference only).

This module provides efficient GPU inference for NeuMIP networks by leveraging
DrJit's cooperative vector API.

Note: CUDA/OptiX cooperative vectors require FP16 internally for optimal layout.
      This module uses FP16 for network weights/computation but accepts Float32
      inputs and returns Float32 outputs.

References:
    - https://drjit.readthedocs.io/en/v1.1.0/coop_vec.html
    - https://drjit.readthedocs.io/en/latest/nn.html
"""

import drjit as dr
import drjit.nn as nn
import torch

from drjit.cuda.ad import Float, Float16, Array2f, Array3f, TensorXf16

def eval_latents(latent_textures, si, active):
    """Evaluate latent textures at the given surface interaction."""
    return [tex.eval_3(si, active) for tex in latent_textures]

def build_neumip_mlp(
    is_offset: bool,
    latent_channels: int = 12,
    width: int = 32,
    depth: int = 2
) -> nn.Sequential:
    """
    Build a NeuMIP MLP network using DrJit's nn module.
    
    Args:
        is_offset: If True, builds offset network (2 + latents -> 1).
                   If False, builds color network (4 + latents -> 3).
        latent_channels: Number of latent channels.
        width: Hidden layer width.
        depth: Number of hidden layers.
    
    Returns:
        An unallocated nn.Sequential network.
    """
    addl_in_channels = 2 if is_offset else 4
    out_channels = 1 if is_offset else 3
    in_channels = latent_channels + addl_in_channels
    
    # Cast to FP16 at input, back to Float at output (required for CUDA coop vectors)
    layers = [nn.Cast(Float16), nn.Linear(in_channels, width)]
    
    for _ in range(depth):
        layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(width, width))
    
    layers.append(nn.LeakyReLU())
    layers.append(nn.Linear(width, out_channels))
    
    # Main MLP has final LeakyReLU, offset MLP does not
    if not is_offset:
        layers.append(nn.LeakyReLU())
    layers.append(nn.Cast(Float))
    return nn.Sequential(*layers)


class NeuMIPCoopMLP:
    """
    DrJit cooperative vector MLP wrapper for inference.
    Uses FP16 internally (required by CUDA/OptiX coop vectors).
    """
    
    def __init__(
        self,
        is_offset: bool,
        latent_channels: int = 12,
        width: int = 32,
        depth: int = 2
    ):
        self.is_offset = is_offset
        self.latent_channels = latent_channels
        self.addl_in_channels = 2 if is_offset else 4
        self.in_channels = latent_channels + self.addl_in_channels
        self.out_channels = 1 if is_offset else 3
        
        # Build network structure with FP16 (required for CUDA coop vectors)
        self.net = build_neumip_mlp(is_offset, latent_channels, width, depth)
        self.net = self.net.alloc(TensorXf16, self.in_channels)
        
        # Pack for inference (required for coop vectors)
        self.weights, self.packed_net = nn.pack(self.net, layout='inference')
      
    
    def eval(self, x: nn.CoopVec) -> nn.CoopVec:
        """Evaluate the network on a cooperative vector input."""
        return self.packed_net(x)


class NeuMIPCoop:
    """
    NeuMIP model using DrJit cooperative vectors for inference.
    
    Accepts Float32 inputs, uses FP16 internally, returns Float32 outputs.
    Assumes latents are pre-sampled and passed directly to eval().
    """
    
    def __init__(
        self,
        use_offset: bool = True,
        latent_channels: int = 12,
        uv_scale: float = 1.0,
        network_width: int = 32,
        network_depth: int = 2,
        min_z: float = 0.6,
        return_offset: bool = False
    ):
        self.use_offset = use_offset
        self.latent_channels = latent_channels
        self.uv_scale = uv_scale
        self.min_z = min_z
        self.return_offset = return_offset
        # Main color MLP
        self.mlp = NeuMIPCoopMLP(
            is_offset=False,
            latent_channels=latent_channels,
            width=network_width,
            depth=network_depth
        )
        
        # Offset MLP (optional)
        self.offset_mlp = None
        if self.use_offset:
            self.offset_mlp = NeuMIPCoopMLP(
                is_offset=True,
                latent_channels=latent_channels,
                width=network_width,
                depth=network_depth
            )
    
    def eval(
        self,
        wi: Array3f,
        wo: Array3f,
        uv: Array2f,
        latent_textures: list,
        offset_latent_textures: list,
        si, active
    ) -> tuple:
        """
        Evaluate the NeuMIP model.
        
        Args:
            wi: Incident light direction (3D, Float32).
            wo: Outgoing/camera direction (3D, Float32).
            latent_textures: List of latent texture objects.
            offset_latent_textures: List of offset latent texture objects.
            si: Surface interaction.
            active: Active mask.
        
        Returns:
            color: RGB color (Float32).
        """
        wo_x, wo_y, wo_z = wo.x, wo.y, wo.z
        wi_x, wi_y = wi.x, wi.y
        
        # Apply UV scale if set
        uv_x, uv_y = uv.x, uv.y
        
        si.uv = uv
        
        if self.use_offset:
            # Sample offset latents
            offset_latents = eval_latents(offset_latent_textures, si, active)

            # Offset network input: [wo.xy, offset_latents (flattened xyz)]
            offset_components = [wo_x, wo_y]
            for lat in offset_latents:
                offset_components.extend([lat.x, lat.y, lat.z])
            offset_input = nn.CoopVec(*offset_components)

            offset_out = self.offset_mlp.eval(offset_input)
            r, = offset_out
            r = Float(r)
            
            # Compute shift: r / clamp(z, min_z) * wo.xy
            z_clamped = dr.maximum(wo_z, self.min_z)
            shift_x = r / z_clamped * wo_x
            shift_y = r / z_clamped * wo_y

            # Apply UV scale to shift if set
            shift_x = shift_x / self.uv_scale
            shift_y = shift_y / self.uv_scale

            if self.return_offset:
                return Array3f(shift_x, shift_y, shift_y*0)
        
            si.uv = Array2f(uv_x + shift_x, uv_y + shift_y)        
        
        # Sample main latents at (possibly shifted) UV
        latents = eval_latents(latent_textures, si, active)

        # Main MLP input: [wi.xy, wo.xy, latents (flattened xyz)]
        main_components = [wi_x, wi_y, wo_x, wo_y]
        for lat in latents:
            main_components.extend([lat.x, lat.y, lat.z])
        main_input = nn.CoopVec(*main_components)
        
        color_out = self.mlp.eval(main_input)
        r, g, b = color_out
        
        color = Array3f(Float(r), Float(g), Float(b))
        
        return color
    
    def eval_color_only(
        self,
        wi: Array3f,
        wo: Array3f,
        latents: list
    ) -> Array3f:
        """
        Evaluate only the color network (skip offset).
        """
        main_input = nn.CoopVec(
            wi.x, wi.y,
            wo.x, wo.y,
            *latents
        )
        color_out = self.mlp.eval(main_input)
        r, g, b = color_out
        
        return Array3f(Float(r), Float(g), Float(b))


def load_torch_weights_to_coop_mlp(coop_mlp: NeuMIPCoopMLP, state_dict: dict, prefix: str):
    """
    Load PyTorch weights into a DrJit cooperative vector MLP.
    
    Dynamically detects linear layer indices from the checkpoint.
    PyTorch uses even indices (0, 2, 4, ...) for Linear layers.
    DrJit uses indices offset by 1 (1, 3, 5, ...) due to initial Cast layer.
    
    Args:
        coop_mlp: The NeuMIPCoopMLP to load weights into.
        state_dict: PyTorch state dict.
        prefix: Key prefix (e.g., 'mlp.network' or 'offset_mlp.network').
    """
    # Dynamically find all Linear layer indices in the checkpoint
    torch_linear_indices = sorted([
        int(k.split('.')[2]) 
        for k in state_dict.keys() 
        if k.startswith(f"{prefix}.") and k.endswith('.weight')
    ])
    
    # DrJit indices are offset by 1 due to initial Cast layer
    drjit_linear_indices = [i + 1 for i in range(0, len(torch_linear_indices) * 2, 2)]
    
    for torch_idx, drjit_idx in zip(torch_linear_indices, drjit_linear_indices):
        weight_key = f"{prefix}.{torch_idx}.weight"
        bias_key = f"{prefix}.{torch_idx}.bias"
        
        if weight_key not in state_dict:
            continue
            
        torch_weight = state_dict[weight_key].numpy()
        torch_bias = state_dict[bias_key].numpy()
        
        linear_layer = coop_mlp.net[drjit_idx]
        
        # Convert to FP16 tensors
        linear_layer.weights = TensorXf16(torch_weight.astype('float16'))
        linear_layer.bias = TensorXf16(torch_bias.astype('float16'))
    
    # Repack for inference after loading weights
    coop_mlp.weights, coop_mlp.packed_net = nn.pack(coop_mlp.net, layout='inference')


def load_checkpoint(coop_model: NeuMIPCoop, checkpoint_path: str):
    """
    Load a PyTorch NeuMIP checkpoint into a NeuMIPCoop model.
    
    Args:
        coop_model: The NeuMIPCoop model to load weights into.
        checkpoint_path: Path to the .pth checkpoint file.
    """
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    # Load main MLP weights
    load_torch_weights_to_coop_mlp(coop_model.mlp, state_dict, 'mlp.network')
    
    # Load offset MLP weights if present
    if coop_model.use_offset and coop_model.offset_mlp is not None:
        load_torch_weights_to_coop_mlp(coop_model.offset_mlp, state_dict, 'offset_mlp.network')
