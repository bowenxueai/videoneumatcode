import torch
import torch.nn as nn
import json
from einops import rearrange
from src.models.textures import LatentTexture, NeuMIPStandaloneLatents, WANLRMTexture, SimpleLRMTexture
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')

class NeuMIPMLP(nn.Module):
    def __init__(self, is_offset, latent_channels = 12, width=32, depth=2) -> None:
        super().__init__()
        addl_in_channels = 4 if not is_offset else 2
        out_channels = 3 if not is_offset else 1
        in_channels = latent_channels + addl_in_channels

        layers =[
                    nn.Linear(in_channels, width), 
                    *[l for _ in range(depth) for l in (nn.LeakyReLU(inplace=True), nn.Linear(width, width))],
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(width, out_channels), 
                ]
        if not is_offset:
            layers += [nn.LeakyReLU(inplace=True)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


    

def my_grid_sample(texture: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """
    Sample the texture at normalized uv in [0,1]^2 using bilinear interpolation.
    Assumption on the shape of inputs:
        texture: [B, C, H, W]
        uv: [B, S, H*, W*, 2] or [B, H*, W*, 2] in [0,1]^2

        the B must match between texture and uv, but H and H* and W and W* can be different. S is another batch dimension for multiple spp.
    """
    new_uv = torch.zeros_like(uv)
    new_uv[..., 0] = uv[..., 0] 
    new_uv[..., 1] = 1 - uv[..., 1]          # flip the y coordinate
    # print('uv lookup before flatten', new_uv.shape)
    _, S, H, _, _  = new_uv.shape
    new_uv = new_uv.flatten(start_dim = 1, end_dim = 2)
    # print('uv lookup after flatten', new_uv.shape)
    # print('grid sample texture', texture.shape)

    out =  torch.nn.functional.grid_sample(texture, grid = 2*new_uv - 1, mode = 'bilinear', align_corners = False)
    # print('grid sample out', out.shape)

    out = out.unflatten(2, (S, H))
    # print('unflattened', out.shape)

    out = rearrange(out, 'b c s h w -> b s h w c')
    # print('rearranged', out.shape)

    return out

def clamp_polar_z(wo: torch.Tensor, z_min: float, eps: float = 1e-8):
    x, y, z = wo.unbind(dim=-1)
    z_min = torch.tensor(z_min).to(wo.device)
    mask = z < z_min

    # radial component in xy-plane
    r = torch.sqrt(x * x + y * y)

    # target xy radius for z = z_min
    r_new = torch.sqrt(1.0 - z_min * z_min)

    # scale factor to preserve azimuth
    scale = r_new / r

    x_new = torch.where(mask, x * scale, x)
    y_new = torch.where(mask, y * scale, y)
    z_new = torch.where(mask, torch.full_like(z, z_min), z)

    wo_clamped = torch.stack((x_new, y_new, z_new), dim=-1)

    return wo_clamped

class NeuMIP(nn.Module):
    # the torch model for NeuMIP
    def __init__(self, use_offset = True, resolution = 1024, latent_channels = 12, uv_scale=1, network_width=32, network_depth=2, min_z=0.6, max_offset=None) -> None:
        super().__init__()
        self.use_offset = use_offset
        self.uv_scale = uv_scale
        self.min_z = min_z
        self.max_offset = max_offset
        self.mlp = NeuMIPMLP(is_offset = False, latent_channels = latent_channels, width = network_width, depth = network_depth)

        if self.use_offset:
            self.offset_mlp = NeuMIPMLP(is_offset = True, latent_channels = latent_channels, width = network_width, depth = network_depth)

    
    def forward(self, wi, wo, uv, latent_texture, offset_texture, chunk_size=None):
        # NOTE: Important assumption: wo is the camera direction at first bounce. In mitsuba, wo is the sampled direction.
        
        # For high-resolution rendering (e.g., 4096x4096), process in spatial tiles
        # to avoid OOM from large intermediate tensors in grid_sample
        B, S, H, W, _ = wi.shape
        total_pixels = H * W
        tile_threshold = 1024 * 1024  # 1M pixels - trigger tiling above this
        
        if total_pixels > tile_threshold:
            return self._forward_tiled(wi, wo, uv, latent_texture, offset_texture)
        else:
            return self._forward_full(wi, wo, uv, latent_texture, offset_texture)
    
    def _forward_full(self, wi, wo, uv, latent_texture, offset_texture):
        """Original forward pass for smaller resolutions."""
        offset_out = torch.zeros_like(wo)


        if self.uv_scale != 1:
            uv *= self.uv_scale

        og_wo = wo.clone()
        z = wo[...,2:3]
        wo = wo[...,0:2]
        wi = wi[...,0:2]

        if self.use_offset:
            offset_latents = my_grid_sample(offset_texture, uv)
            wo_offset = og_wo.clone()
            wo_offset = clamp_polar_z(wo_offset, self.min_z)
            z = wo_offset[..., 2:3]
            x = torch.cat([wo_offset[..., 0:2], offset_latents], dim = -1)
            r = self.offset_mlp(x)
            z = torch.clamp(z, min = self.min_z)
            shift = r / z*(wo_offset[..., 0:2])
            if self.max_offset is not None:
                shift = shift.clamp(max=self.max_offset, min=-self.max_offset)

            if self.uv_scale != 1:
                shift *= self.uv_scale
            uv1 = uv +  shift
            offset_out =  torch.cat([shift, torch.zeros_like(z)], dim = -1)

        latents = my_grid_sample(latent_texture, uv1 if self.use_offset else uv)
        x = torch.cat([wi, wo, latents], dim = -1)
        
        out = self.mlp(x)
        return out, offset_out


    def _forward_tiled(self, wi, wo, uv, latent_texture, offset_texture):
        """Tiled forward pass for high-resolution rendering to avoid OOM."""
        B, S, H, W, _ = wi.shape
        
        # Tile size: process 64 rows at a time
        tile_size = 64
        
        # Pre-allocate outputs
        out = torch.empty(B, S, H, W, 3, device=wi.device, dtype=wi.dtype)
        offset_out = torch.empty(B, S, H, W, 3, device=wi.device, dtype=wi.dtype)
        
        for h_start in range(0, H, tile_size):
            h_end = min(h_start + tile_size, H)
            
            # Extract tile
            wi_tile = wi[:, :, h_start:h_end, :, :]
            wo_tile = wo[:, :, h_start:h_end, :, :]
            uv_tile = uv[:, :, h_start:h_end, :, :]
            
            # Use gradient checkpointing to trade compute for memory
            # This recomputes forward during backward instead of saving activations
            if wi.requires_grad or latent_texture.requires_grad:
                out_tile, offset_tile = torch.utils.checkpoint.checkpoint(
                    self._forward_full,
                    wi_tile, wo_tile, uv_tile, latent_texture, offset_texture,
                    use_reentrant=False
                )
            else:
                out_tile, offset_tile = self._forward_full(
                    wi_tile, wo_tile, uv_tile, latent_texture, offset_texture
                )
            
            # Store results
            out[:, :, h_start:h_end, :, :] = out_tile
            offset_out[:, :, h_start:h_end, :, :] = offset_tile
            
            # Free intermediate memory
            del wi_tile, wo_tile, uv_tile, out_tile, offset_tile
            torch.cuda.empty_cache()
        
        return out, offset_out

    def set_trainable(self, trainable: bool):
        [p.requires_grad_(trainable) for p in self.parameters()]


def initalize_network(desc):
    return NeuMIP(**desc)


def initalize_latents(type, desc) -> LatentTexture:
    if type.lower() == 'standalone':
        return NeuMIPStandaloneLatents(**desc)
    elif type.lower() == 'wan_lrm':
        return WANLRMTexture(**desc)
    elif type.lower() == 'simple_lrm':
        return SimpleLRMTexture(**desc)
    else:
        raise ValueError(f"Invalid latent texture type: {type}")
