from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
import imageio.v3 as iio
import numpy as np


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Convert sRGB in [0,1] to linear RGB."""
    a = 0.055
    low = x <= 0.04045
    high = ~low
    out = torch.empty_like(x)
    out[low] = x[low] / 12.92
    out[high] = ((x[high] + a) / (1 + a)) ** 2.4
    return out


def _to_float_tensor(image: np.ndarray, device: torch.device, srgb: bool) -> torch.Tensor:
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32) / 255.0
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    tex = torch.from_numpy(image).to(device)
    if srgb:
        tex = _srgb_to_linear(tex.clamp(0.0, 1.0))
    return tex


@dataclass
class Texture:
    texture: torch.Tensor  # (H, W, 3) in [0,1]
    device: torch.device

    @staticmethod
    def from_file(texture_path: str, device: torch.device | str = "cpu", srgb: bool = True) -> "TexturedPlane":
        device = torch.device(device)
        image = iio.imread(texture_path)
        tex = _to_float_tensor(image, device, srgb=srgb)
        return Texture(
            texture=tex,
            device=device,
        )

    def grid_sample(self, uv: torch.Tensor) -> torch.Tensor:
        """Sample the texture at normalized uv in [0,1]^2 using bilinear interpolation.

        UV convention: (0,0) is bottom-left; (1,1) is top-right.
        Uses torch.grid_sample under the hood.

        Args:
            uv: (H, W, 2) or (S, H, W, 2)
        Returns:
            color: (H, W, 3) or (S, H, W, 3)
        """
        tex = self.texture  # (Ht, Wt, 3)

        # Determine batch dims (e.g., S) and image size
        if uv.dim() == 3:
            batch_shape = ()
            H, W = uv.shape[0], uv.shape[1]
            grid = uv
        elif uv.dim() == 4:
            batch_shape = uv.shape[:-3]
            H, W = uv.shape[-3], uv.shape[-2]
            grid = uv
        else:
            raise ValueError("uv must have shape (H,W,2) or (S,H,W,2)")

        # Build grid_sample grid from uv in [0,1]^2 to [-1,1]^2
        u = grid[..., 0].clamp(0.0, 1.0)
        v = grid[..., 1].clamp(0.0, 1.0)
        x = u * 2.0 - 1.0
        y = v * 2.0 - 1.0
        grid_n = torch.stack([x, y], dim=-1)  # (..., H, W, 2)

        # Prepare texture as NCHW and repeat across batch
        tex_nchw = tex.permute(2, 0, 1).unsqueeze(0)  # (1, 3, Ht, Wt)
        batch = int(np.prod(batch_shape)) if len(batch_shape) > 0 else 1
        tex_nchw = tex_nchw.expand(batch, -1, -1, -1)  # (B,3,Ht,Wt)

        # Flatten batch dims for grid_sample
        grid_flat = grid_n.reshape(batch, H, W, 2)

        sampled = F.grid_sample(
            tex_nchw,
            grid_flat,
            mode="bilinear",
            align_corners=False,
        )  # (B, 3, H, W)

        sampled = sampled.permute(0, 2, 3, 1)  # (B, H, W, 3)
        if len(batch_shape) == 0:
            return sampled.squeeze(0)
        else:
            return sampled.reshape(*batch_shape, H, W, 3)


