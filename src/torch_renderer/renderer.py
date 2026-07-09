from __future__ import annotations

from typing import Tuple, Dict

import torch
import math
import torch.nn.functional as F

from .camera import PerspectiveCamera
from .geometry import Plane
from .brdf import DiffuseBRDF
from src.neumip_nets import NeuMIP


def intersect_plane(
    origins: torch.Tensor,
    directions: torch.Tensor,
    plane_point: torch.Tensor,
    plane_normal: torch.Tensor,
    plane_width: float | None = None,
    plane_height: float | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ray-plane intersection.

    Args:
        origins: (H, W, 3)
        directions: (H, W, 3) normalized
        plane_point: (3,)
        plane_normal: (3,) normalized
    Returns:
        hit_mask: (H, W, 1) bool
        points: (H, W, 3)
    """
    denom = (directions * plane_normal).sum(dim=-1, keepdim=True)
    parallel = denom.abs() < 1e-8
    t = ((plane_point - origins) * plane_normal).sum(dim=-1, keepdim=True) / torch.where(parallel, torch.ones_like(denom), denom)
    hit_mask = (~parallel) & (t > 1e-6)
    points = origins + t * directions

    # If finite plane dimensions are provided, clip intersections to the plane extent
    if plane_width is not None and plane_height is not None:
        half_w = float(plane_width) * 0.5
        half_h = float(plane_height) * 0.5
        in_x = points[..., 0].abs() <= half_w
        in_y = points[..., 1].abs() <= half_h
        hit_mask = hit_mask & in_x[..., None] & in_y[..., None]

    # UVs in [0,1]
    u = points[..., 0] / plane_width + 0.5
    v = points[..., 1] / plane_height + 0.5
    uv = torch.stack([u, v], dim=-1)  # (S,H,W,2)

    return hit_mask, points, uv


def _gaussian_kernel_1d(stddev: float, radius: float, device: torch.device) -> torch.Tensor:
    k = int(2 * math.ceil(radius) + 1)
    half = k // 2
    x = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / float(stddev)) ** 2)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, k)


def _gaussian_blur(image: torch.Tensor, stddev: float, radius: float) -> torch.Tensor:
    """Apply separable Gaussian blur to image (..., H, W, 3) in linear space.

    Supports optional leading batch dimensions (e.g., B) and leaves them intact.
    """
    device = image.device
    C = image.shape[-1]
    H, W = image.shape[-3], image.shape[-2]
    leading_shape = image.shape[:-3]
    leading_num = int(torch.tensor(leading_shape).prod().item()) if len(leading_shape) > 0 else 1

    kernel_1d = _gaussian_kernel_1d(stddev, radius, device)  # (1,1,k)

    # Prepare weights for depthwise conv
    k = kernel_1d.shape[-1]
    weight_h = kernel_1d.view(1, 1, k, 1).repeat(C, 1, 1, 1)  # (C,1,k,1)
    weight_w = kernel_1d.view(1, 1, 1, k).repeat(C, 1, 1, 1)  # (C,1,1,k)

    # Flatten leading dims for conv
    x = image.reshape(leading_num, H, W, C).permute(0, 3, 1, 2)  # (Bf,C,H,W)
    x = F.conv2d(x, weight_h, padding=(k // 2, 0), groups=C)
    x = F.conv2d(x, weight_w, padding=(0, k // 2), groups=C)
    x = x.permute(0, 2, 3, 1).reshape(*leading_shape, H, W, C)
    return x


def render(
    camera: PerspectiveCamera,
    plane: Plane,
    brdf: DiffuseBRDF,
    image_size: Tuple[int, int],
    light_position: torch.Tensor,
    light_intensity: torch.Tensor | float = 1.0,
    apply_filter: bool = True,
    filter_stddev: float = 0.5,
    filter_radius: float = 2.0,
    spp: int = 1,
    jitter: bool = True,
    seed: int | None = None,
    latent_texture: torch.Tensor | None = None,
    offset_texture: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """Render a textured plane with direct illumination (Lambertian), no GI.

    Args:
        camera: pinhole camera
        plane: textured plane at z=0 (by default)
        brdf: diffuse BRDF
        image_size: (height, width)
        light_position: (3,) tensor on same device
        light_intensity: (3,) or scalar
    Returns:
        image: (H, W, 3) in [0,1]
    """
    device = camera.device
    H, W = image_size

    # Rays: (S,H,W,3) if unbatched camera, or (B,S,H,W,3) if batched camera
    origins, directions = camera.generate_rays(H, W, spp=spp, jitter=jitter, seed=seed)
    # Intersections
    hit_mask, points, uv = intersect_plane(
        origins,
        directions,
        plane.plane_point,
        plane.plane_normal,
        plane_width=plane.width,
        plane_height=plane.height,
    )

    wo = -directions



    # Lighting (support batched or single light positions)
    rays_ndim = origins.dim()  # 4 (S,H,W,3) or 5 (B,S,H,W,3)
    if torch.is_tensor(light_position):
        lp = light_position.to(device)
        if rays_ndim == 5:
            B = origins.shape[0]
            if lp.dim() == 2:
                # (B,3)
                Lpos = lp.view(B, 1, 1, 1, 3)
            else:
                # (3,)
                Lpos = lp.view(1, 1, 1, 1, 3)
        else:
            # Unbatched camera
            Lpos = lp.view(1, 1, 1, 3)
    else:
        # Scalar/list input -> treat as single light
        Lpos = torch.as_tensor(light_position, dtype=torch.float32, device=device).view(1, *( [1] if origins.dim()==5 else [] ), 1, 1, 3) if origins.dim()==5 else torch.as_tensor(light_position, dtype=torch.float32, device=device).view(1,1,1,3)

    to_light = Lpos - points
    dist2 = (to_light * to_light).sum(dim=-1, keepdim=True).clamp_min(1e-8)
    wi = torch.nn.functional.normalize(to_light, dim=-1, eps=1e-8)

    # Plane normal broadcasts over batch/sample/spatial dims
    n = plane.plane_normal.view(*([1] * (origins.dim() - 1)), 3)
    ndotl = (n * wi).sum(dim=-1, keepdim=True).clamp(min=0.0)



    # BRDF
    if True:
        fr, brdf_offset = brdf(wi, wo,uv, latent_texture, offset_texture)
    else:
        fr = brdf(wi, wo,uv)
        brdf_offset = torch.zeros_like(wo)

    # Light intensity/color
    if not torch.is_tensor(light_intensity):
        light_color = torch.tensor([light_intensity, light_intensity, light_intensity], dtype=torch.float32, device=device)
    else:
        light_color = light_intensity.to(device)
    if origins.dim() == 5:
        if light_color.dim() == 2:
            # (B,3)
            light_color = light_color.view(origins.shape[0], 1, 1, 1, 3)
        else:
            light_color = light_color.view(1, 1, 1, 1, 3)
    else:
        light_color = light_color.view(1, 1, 1, 3)

    Li = light_color / dist2
    color_s = fr * ndotl * Li

    # Apply visibility
    color_s = torch.where(hit_mask, color_s, torch.zeros_like(color_s))

    # Means across samples (S dimension is always the 4th from the end)
    sample_dim = -4
    color = color_s.mean(dim=sample_dim)
    vis = hit_mask.float().mean(dim=sample_dim)

    # Visibility-weighted aux maps (average over hits across SPP)
    # eps = 1e-8
    # hit_count = hit_mask.float().sum(dim=0)  # (H,W,1)
    # denom = torch.clamp(hit_count, min=eps)
    uv_map = (hit_mask.float() * uv).mean(dim=sample_dim)
    cam_dir_map = (hit_mask.float() * directions).mean(dim=sample_dim)
    light_dir_map = (hit_mask.float() * wi).mean(dim=sample_dim)
    albedo_reflectance = (hit_mask.float() * (fr * torch.pi)).mean(dim=sample_dim)
    brdf_offset = (hit_mask.float() * (brdf_offset)).mean(dim=sample_dim)


    result =  {
        "brdf_offset": brdf_offset,
        "image": color,  # linear radiance
        "visibility": vis,
        "uv": uv_map,
        "albedo": albedo_reflectance,
        "cam_dir": cam_dir_map,
        "light_dir": light_dir_map,
    }

    # Optional Gaussian reconstruction filter (Mitsuba default)
    if apply_filter:
        result = {key: _gaussian_blur(value, stddev=float(filter_stddev), radius=float(filter_radius)) for key, value in result.items()}

    return result

