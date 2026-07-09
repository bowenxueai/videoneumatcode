import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from typing import Tuple, Optional
from .wan_video_dit import Head
import math
from .wan_video_vae import RMS_norm, CausalConv3d, ResidualBlock, Resample
from torch.utils.checkpoint import checkpoint
from accelerate.logging import get_logger
logger = get_logger(__name__)


class VAEAddonUpsampler(nn.Module):
    """
    Configurable spatial upsampler following exact WAN VAE decoder architecture.
    
    Designed to be used in nn.Sequential after RMS_norm + SiLU:
        self.vae.decoder.head = nn.Sequential(
            RMS_norm(96, images=False),
            nn.SiLU(),
            VAEAddonUpsampler(in_channels=96, out_channels=24, num_upsample_stages=2),
        )
    
    Architecture matches existing VAE decoder pattern:
        - For each stage: Resample (2x spatial, halves channels) + N ResidualBlocks
        - CausalConv3d (output projection)
    
    Args:
        num_upsample_stages: Number of 2x upsample stages (1=2x, 2=4x, 3=8x, etc.)
    
    Input:  [B, 96, T, H, W]   (from VAE decoder after SiLU, at 8x)
    Output: [B, out_channels, T, H * 2^num_upsample_stages, W * 2^num_upsample_stages]
    """
    
    def __init__(
        self, 
        in_channels: int = 96,   # matches last VAE decoder stage
        out_channels: int = 24,  # latent_channels * 2 (12 * 2)
        num_res_blocks: int = 3, # matches existing VAE pattern
        num_upsample_stages: int = 2,  # 1=2x, 2=4x, 3=8x upsampling
        temporal_upsample: list = None,  # [False, False, ...] per stage
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,  # Enable gradient checkpointing to save memory
    ):
        super().__init__()
        self.num_upsample_stages = num_upsample_stages
        self.gradient_checkpointing = gradient_checkpointing
        
        if temporal_upsample is None:
            temporal_upsample = [False] * num_upsample_stages
        
        # Build stages dynamically
        self.upsamples = nn.ModuleList()
        self.stages = nn.ModuleList()
        
        current_dim = in_channels
        for i in range(num_upsample_stages):
            # Upsample: 2x spatial, halves channels
            mode = 'upsample3d' if temporal_upsample[i] else 'upsample2d'
            self.upsamples.append(Resample(current_dim, mode=mode))
            next_dim = current_dim // 2
            if num_upsample_stages == 2 and i == 1:
                self.upsamples[-1].resample[1] = nn.Conv2d(current_dim, current_dim, 3, padding=1)
                next_dim = current_dim
            
            # ResidualBlocks at this dimension
            stage_blocks = nn.ModuleList([
                ResidualBlock(next_dim, next_dim, dropout)
                for _ in range(num_res_blocks)
            ])
            self.stages.append(stage_blocks)
            current_dim = next_dim
        
        # Output projection (like VAE decoder head)
        self.conv_out = CausalConv3d(current_dim, out_channels, 3, padding=1)
    
    def _run_block(self, block, x):
        """Helper to run a single block, used for checkpointing."""
        return block(x)
    
    def forward(self, x):
        """
        Args:
            x: [B, in_channels, T, H, W] from VAE decoder (after RMS_norm + SiLU)
        Returns:
            [B, out_channels, T, H * 2^num_stages, W * 2^num_stages]
        """
        for upsample, stage_blocks in zip(self.upsamples, self.stages):
            # Upsample: 2x spatial, halves channels
            x = upsample(x)
            
            # ResidualBlocks
            for block in stage_blocks:
                if self.gradient_checkpointing and self.training:
                    x = checkpoint(self._run_block, block, x, use_reentrant=False)
                else:
                    x = block(x)
        
        # Output projection
        x = self.conv_out(x)
        
        return x
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False





class DefaultHead(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float, latent_channels: int, factor: int, wan_dit_head_weights: list[str] = None, **kwargs):
        super().__init__()
        self.head = Head(dim, out_dim, patch_size, eps)
        if wan_dit_head_weights is not None:
            result = self.head.load_state_dict(wan_dit_head_weights)
            logger.info(f"Wan dit head weights attempted to load, result: {result}")


        else:
            logger.info(f"Default head with re-initialized params")

        self.factor = factor
        self.patch_size = patch_size

    def forward(self, x, t, grid_size: Tuple[int, int, int]):
        f, h, w = grid_size
        x = self.head(x, t)

        # unpatchify the latent tokens
        x = rearrange(
                    x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
                    f=f, h=h, w=w, 
                    x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
                )        
        return x


# my linear upsampler

class LinearUpSample(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float, latent_channels: int, factor: int, **kwargs):
        super().__init__()
        self.head = Head(dim, factor * factor * 2 * latent_channels, patch_size, eps) # 2 is for latents & offset latents
        self.factor = factor
        self.latent_channels = latent_channels
        self.patch_size = patch_size

    def forward(self, x, t, grid_size: Tuple[int, int, int]):
        f, h, w = grid_size
        x = self.head(x, t)

        # unpatchify the latent tokens
        x = rearrange(
                    x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
                    c= self.latent_channels*2, f=f, h=h, w=w,
                    x=self.patch_size[0], y=self.factor*self.patch_size[1], z=self.factor*self.patch_size[2])
        return x


class LinearConvUpSample(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float, latent_channels: int, factor: int, **kwargs):
        super().__init__()
        self.head = Head(dim, factor * factor * 2 * latent_channels, patch_size, eps) # 2 is for latents & offset latents
        self.factor = factor
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.conv_out = nn.Conv2d(latent_channels * 2, latent_channels * 2, kernel_size=3, padding = 1)
        # self._init_gaussian_blur(self.conv_out, sigma=1.0)
    
    def _init_gaussian_blur(self, conv: nn.Conv2d, sigma: float = 1.0):
        """Initialize a Conv2d to act as a Gaussian blur filter."""
        kernel_size = conv.kernel_size[0]
        channels = conv.in_channels
        
        x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        
        with torch.no_grad():
            conv.weight.zero_()
            for i in range(channels):
                conv.weight[i, i] = gauss_2d
            if conv.bias is not None:
                conv.bias.zero_()

    def forward(self, x, t, grid_size: Tuple[int, int, int]):
        f, h, w = grid_size
        x = self.head(x, t)

        # unpatchify the latent tokens
        x = rearrange(
                    x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
                    c= self.latent_channels*2, f=f, h=h, w=w,
                    x=self.patch_size[0], y=self.factor*self.patch_size[1], z=self.factor*self.patch_size[2])
        b = x.shape[0]
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        x = self.conv_out(x)
        x = rearrange(x, '(b f) c h w -> b c f h w', b=b)
        return x


# my convolution upsampler

class ResidualUpBlock(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()

        self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode="nearest")

        # Conv → GN(4) → SiLU
        self.conv1 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(4, out_dim)
        self.act1 = nn.SiLU()

        # Conv → GN(4)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(4, out_dim)


        self.act_out = nn.SiLU()

    def forward(self, x):
        # Upsample first
        x = self.upsample(x)

        # Residual path
        skip = x

        # Conv stack
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        # Residual add
        out = out + skip

        # Final activation
        out = self.act_out(out)

        return out
    
    # def initialize(self, device="cpu", dtype=torch.float32):
    #     """Materialize and initialize ONLY the meta parameters in this ResUp block."""

    #     # ------- conv1 -------
    #     if any(p.is_meta for p in self.conv1.parameters(recurse=False)):
    #         self.conv1.to_empty(device=device)
    #         self.conv1.reset_parameters()

    #     # ------- conv2 -------
    #     if any(p.is_meta for p in self.conv2.parameters(recurse=False)):
    #         self.conv2.to_empty(device=device)
    #         self.conv2.reset_parameters()

    #     # ------- norm1 -------
    #     if any(p.is_meta for p in self.norm1.parameters(recurse=False)):
    #         self.norm1.to_empty(device=device)
    #         # GroupNorm.reset_parameters() exists and sets weight=1, bias=0
    #         self.norm1.reset_parameters()

    #     # ------- norm2 -------
    #     if any(p.is_meta for p in self.norm2.parameters(recurse=False)):
    #         self.norm2.to_empty(device=device)
    #         self.norm2.reset_parameters()


class MyHybridUpSample(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float, latent_channels: int, factor: int, wan_dit_head_weights=None, **kwargs):
        super().__init__()
        self.dim = dim
        # self.out_dim = latent_channels * 2 # not out_dim since it is the VAE latenst channel size = 16
        self.head = DefaultHead(dim, latent_channels * 2, patch_size, eps, latent_channels, factor, wan_dit_head_weights=wan_dit_head_weights) # the default head of DiT, scales h,w by 2 already (for unpatchifying)
        self.patch_size = patch_size
        self.upsample = nn.Sequential(
            *[ResidualUpBlock(latent_channels * 2 ) for _ in range(int(math.log2(factor)))],
        )
        self.latent_channels = latent_channels
        self.factor = factor
        self.conv_out = nn.Conv2d(latent_channels * 2, latent_channels * 2, kernel_size=1)
        # Zero-initialize for stable start (same as ConvHead)
        # nn.init.zeros_(self.conv_out.weight)
        # nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, t, grid_size: Tuple[int, int, int]):
        x = self.head(x, t, grid_size)
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        x = self.upsample(x)
        x = self.conv_out(x)
        x = rearrange(x, '(b f) c h w -> b c f h w', b=b)

        return x


    # def initialize(self, device="cpu", dtype=torch.float32):
    #     """Materialize and initialize this layer after checkpoint loading."""
    #     # --- Materialize + initialize Linear ---
    #     if any(p.is_meta for p in self.parameters(recurse=False)):
    #         self.to_empty(device=device)  # allocate real tensors
    #         self.reset_parameters()                    # now fills with defaults

    #     # --- Initialize each ResidualUpBlock16 inside self.upsample ---
    #     for block in self.upsample:
    #         block.initialize(device=device, dtype=dtype)


    #     if any(p.is_meta for p in self.conv_out.parameters(recurse=False)):
    #         self.conv_out.to_empty(device=device)
    #         self.conv_out.reset_parameters()

        # # --- Initialize modulation manually ---
        # if self.modulation.is_meta:
        #     dim = self.modulation.shape[-1]
        #     self.modulation = nn.Parameter(
        #         torch.randn(1, 2, dim, device=device, dtype=dtype) / dim**0.5
        #     )



class ConvHead(nn.Module):
    """
    Convolutional head for WAN DiT that replaces linear projection.
    Uses convolutions with spatial upsampling (16x for h,w) to avoid patch artifacts.
    
    The original head does: [B, N, dim] -> [B, N, 1*16*16*24] then unpatchify
    This does: [B, N, dim] -> reshape to [B, dim, f, h, w] -> conv upsample -> [B, 24, f, h*16, w*16]
    """
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float, latent_channels: int, factor: int, min_channels: int = 64, **kwargs):
        super().__init__()
        self.dim = dim
        # self.out_dim = out_dim # this is the vae latent channels, but we perhaps do not need to use this here
        self.latent_channels = latent_channels
        self.out_dim = latent_channels * 2
        self.factor = factor
        self.patch_size = patch_size
        self.spatial_upsample = self.factor * self.patch_size[2]  # 16x for h and w
        assert self.patch_size[0] == 1 and self.patch_size[1] == 2 and self.patch_size[2] == 2
        
        # Modulation (same as original Head for compatibility)
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        # self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        
        # Number of 2x upsample stages needed (16 = 2^4)
        n_upsample = int(math.log2(self.spatial_upsample))
        
        # Build decoder with progressively reducing channels
        # Using 2D convs applied per-frame (since temporal factor is usually 1)
        channel_schedule = []
        current = dim
        for i in range(n_upsample):
            next_dim = max(current // 2, min_channels)
            channel_schedule.append((current, next_dim))
            current = next_dim
        
        layers = []
        for i, (ch_in, ch_out) in enumerate(channel_schedule):
            # Conv -> PixelShuffle(2) for 2x spatial upsampling
            layers.append(nn.Conv2d(ch_in, ch_out * 4, kernel_size=3, padding=1))
            layers.append(nn.PixelShuffle(2))
            if i < n_upsample - 1:
                layers.append(nn.GELU())
        
        # Final projection to output channels
        layers.append(nn.GELU())
        layers.append(nn.Conv2d(channel_schedule[-1][1], self.out_dim, kernel_size=3, padding=1))
        # layers.append(nn.GELU())
        
        # Final 1x1 conv for channel mixing
        final_conv = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1)
        # Zero-initialize to start with stable/zero output (common in diffusion models)
        # nn.init.zeros_(final_conv.weight)
        # nn.init.zeros_(final_conv.bias)
        layers.append(final_conv)
        
        self.decoder = nn.Sequential(*layers)

    def forward(self, x, t, grid_size=None):
        """
        Args:
            x: [B, N, dim] tokens where N = f * h * w
            t: time modulation
            grid_size: (f, h, w) tuple for reshaping
        Returns:
            [B, out_channels, f, h*16, w*16]
        """
        B, N, D = x.shape
        
        # Apply modulation (same as original Head)
        # if len(t.shape) == 3:
        #     shift, scale = (self.modulation.unsqueeze(0).to(dtype=t.dtype, device=t.device) + t.unsqueeze(2)).chunk(2, dim=2)
        #     x = self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)
        # else:
        #     shift, scale = (self.modulation.to(dtype=t.dtype, device=t.device) + t).chunk(2, dim=1)
        #     x = self.norm(x) * (1 + scale) + shift
        x = self.norm(x)
        f, h, w = grid_size
        
        x = rearrange(x, 'b (f h w) d -> (b f) d h w', f=f, h=h, w=w)
        
        # Apply convolutional decoder (2D per frame)
        x = self.decoder(x)  # [(B*f), out_channels, h*16, w*16]
        
        # Reshape back to 5D: [(B*f), C, H, W] -> [B, C, f, H, W]
        x = rearrange(x, '(b f) c h w -> b c f h w', b=B, f=f)
        
        return x

    # def initialize(self, device="cpu", dtype=torch.float32):
    #     """Materialize and initialize this layer after checkpoint loading (for meta tensors)."""
    #     # Initialize LayerNorm if meta
    #     if any(p.is_meta for p in self.norm.parameters(recurse=False)):
    #         self.norm.to_empty(device=device)
    #         self.norm.reset_parameters()
        
    #     # Initialize modulation if meta
    #     if self.modulation.is_meta:
    #         dim = self.modulation.shape[-1]
    #         self.modulation = nn.Parameter(
    #             torch.randn(1, 2, dim, device=device, dtype=dtype) / dim**0.5
    #         )
        
    #     # Initialize decoder conv layers if meta
    #     for module in self.decoder:
    #         if isinstance(module, nn.Conv2d):
    #             if any(p.is_meta for p in module.parameters(recurse=False)):
    #                 module.to_empty(device=device)
    #                 module.reset_parameters()






class VAEDecoderHead(nn.Module):
    """
    Upsampler using LinearHead + VAE decoder (trainable end-to-end).
    
    Flow:
    1. DefaultHead (linear) → [B, out_dim=16, f, h*2, w*2]
    2. VAE decoder (8x) → [B, 3, f, h*16, w*16]
    3. Simple 1x1 conv → [B, latent_channels*2, f, h*16, w*16]
    
    Total upsampling: 2x (unpatchify) * 8x (VAE) = 16x
    With addon_upsampler=True: 2x * 8x * 4x = 64x (but addon does 4x, so 32x total from VAE)
    """
    def __init__(
        self, 
        dim: int, 
        out_dim: int,  # VAE z_dim = 16
        patch_size: Tuple[int, int, int], 
        eps: float, 
        latent_channels: int, 
        factor: int,
        vae_path: str,
        temporal_decoding: bool,
        wan_dit_head_weights = None,
        addon_upsampler: bool = False,  # Enable addon upsampler
        addon_num_upsample_stages: int = 2,  # 1=2x, 2=4x, 3=8x addon upsampling
        addon_num_res_blocks: int = 3,
        addon_temporal_upsample: list = None,
        addon_gradient_checkpointing: bool = False,  # Enable gradient checkpointing for addon
        vae_gradient_checkpointing: bool = False,  # Enable gradient checkpointing for base VAE decoder
        random_init: bool = False,
        **kwargs,
    ):
        super().__init__()
        from ..wan_lrm import load_LRM_vae
        assert vae_path is not None
        self.latent_channels = latent_channels
        self.addon_upsampler_enabled = addon_upsampler
        self.vae_gradient_checkpointing = vae_gradient_checkpointing
        
        # Linear head (same as MyHybridUpSample)
        self.head = DefaultHead(dim, out_dim, patch_size, eps, latent_channels, factor)
        
        # Load VAE decoder (trainable)
        vae = load_LRM_vae(vae_path, temporal_decoding, random_init)
        self.scale = vae.scale
        del vae.model.encoder
        del vae.model.conv1
        del vae.model.decoder.head
        self.vae = vae.model
        torch.cuda.empty_cache()
        
        # Build decoder head with optional addon upsampler
        if addon_upsampler:
            # Total: 8x (base VAE) * 2^addon_num_upsample_stages (addon)
            self.addon_upsampler_module = VAEAddonUpsampler(
                in_channels=96,
                out_channels=latent_channels * 2,
                num_res_blocks=addon_num_res_blocks,
                num_upsample_stages=addon_num_upsample_stages,
                temporal_upsample=addon_temporal_upsample,
                gradient_checkpointing=addon_gradient_checkpointing,
            )
            self.vae.decoder.head = nn.Sequential(
                RMS_norm(96, images=False),
                nn.SiLU(),
                self.addon_upsampler_module,
            )
        else:
            # 8x (original)
            self.addon_upsampler_module = None
            self.vae.decoder.head = nn.Sequential(
                RMS_norm(96, images=False),
                nn.SiLU(),
                CausalConv3d(96, latent_channels * 2, 3, padding=1),
            )
        
        self.temporal_decoding = temporal_decoding
    
    def enable_addon_gradient_checkpointing(self):
        """Enable gradient checkpointing for the addon upsampler."""
        if self.addon_upsampler_module is not None:
            self.addon_upsampler_module.enable_gradient_checkpointing()
    
    def disable_addon_gradient_checkpointing(self):
        """Disable gradient checkpointing for the addon upsampler."""
        if self.addon_upsampler_module is not None:
            self.addon_upsampler_module.disable_gradient_checkpointing()
        
    def _vae_decode(self, x):
        """Helper for VAE decode, used for gradient checkpointing."""
        return self.vae.decode(x, self.scale)
    
    def forward(self, x, t, grid_size: Tuple[int, int, int]):
        B = x.shape[0]
        f, _, _ = grid_size
        
        # Linear head: [B, N, dim] -> [B, 16, f, h*2, w*2]
        x = self.head(x, t, grid_size)
        if not self.temporal_decoding:
            x = x[:,:,0:1]
 
        # VAE decode: [B, 16, f, H, W] -> [B, latent_channels*2, f, H*8, W*8]
        if self.vae_gradient_checkpointing and self.training:
            x = checkpoint(self._vae_decode, x, use_reentrant=False)
        else:
            x = self.vae.decode(x, self.scale)
        x = x[:,:,0:1]
        
        return x


def initialize_upsampler(type: str, **desc):
    if type == "default":
        return DefaultHead(**desc)
    elif type == "linear":
        return LinearUpSample(**desc)
    elif type == "hybrid":
        return MyHybridUpSample(**desc)
    elif type == "conv":
        return ConvHead(**desc)
    elif type == "linear-conv":
        return LinearConvUpSample(**desc)
    elif type == "vae":
        return VAEDecoderHead(**desc)
    else:
        raise ValueError(f"Invalid upsampler type: {type}")