from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class PerspectiveCamera:
    origin: torch.Tensor  # (3,) or (B, 3)
    forward: torch.Tensor  # (3,) or (B, 3)
    right: torch.Tensor  # (3,) or (B, 3)
    up: torch.Tensor  # (3,) or (B, 3)
    fov_y_deg: float | torch.Tensor  # scalar or (B,)
    aspect: float | torch.Tensor  # scalar or (B,)
    device: torch.device

    def generate_rays(self, height: int, width: int, spp: int = 1, jitter: bool = True, seed: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate per-pixel rays for a pinhole camera.

        Args:
            height, width: image dimensions
            spp: samples per pixel (creates an S dimension)
            jitter: if True, sample subpixel offsets in [-0.5, 0.5]
            seed: RNG seed for reproducibility

        Returns:
            origins:
                - (S, H, W, 3) if camera is unbatched
                - (B, S, H, W, 3) if camera parameters are batched with batch size B
            directions: same shape as origins, normalized
        """
        device = self.device
        H, W = height, width
        if seed is not None:
            torch.manual_seed(seed)
        S = max(1, int(spp))
        # Pixel-centered NDC coordinates in [-1, 1] using ij indexing
        jj = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W).expand(1, H, W)
        ii = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1).expand(1, H, W)
        if jitter:
            jitter_x = torch.rand((S, H, W), device=device) - 0.5
            jitter_y = torch.rand((S, H, W), device=device) - 0.5
        else:
            jitter_x = torch.zeros((S, H, W), device=device)
            jitter_y = torch.zeros((S, H, W), device=device)
        xv = ((jj + 0.5 + jitter_x) / float(W)) * 2.0 - 1.0  # (S, H, W)
        yv = ((ii + 0.5 + jitter_y) / float(H)) * 2.0 - 1.0  # (S, H, W)

        # Determine if camera parameters are batched (B,3) or single (3,)
        is_batched = self.origin.dim() == 2

        if is_batched:
            B = self.origin.shape[0]
            xv_b = xv.unsqueeze(0).expand(B, -1, -1, -1)  # (B, S, H, W)
            yv_b = yv.unsqueeze(0).expand(B, -1, -1, -1)  # (B, S, H, W)

            fov_tensor = torch.as_tensor(self.fov_y_deg, dtype=torch.float32, device=device)
            if fov_tensor.dim() == 0:
                fov_tensor = fov_tensor.repeat(B)
            tan_half_fovy = torch.tan(fov_tensor * (torch.pi / 180.0) * 0.5)  # (B,)
            aspect_tensor = torch.as_tensor(self.aspect, dtype=torch.float32, device=device)
            if aspect_tensor.dim() == 0:
                aspect_tensor = aspect_tensor.repeat(B)

            px = xv_b * tan_half_fovy.view(B, 1, 1, 1) * aspect_tensor.view(B, 1, 1, 1)  # (B, S, H, W)
            py = -yv_b * tan_half_fovy.view(B, 1, 1, 1)  # (B, S, H, W)

            forward = self.forward  # (B,3)
            right = self.right      # (B,3)
            up = self.up            # (B,3)

            dirs = (
                px[..., None] * right[:, None, None, None, :] +
                py[..., None] * up[:, None, None, None, :] +
                forward[:, None, None, None, :]
            )  # (B, S, H, W, 3)
            dirs = torch.nn.functional.normalize(dirs, dim=-1, eps=1e-8)

            origins = self.origin[:, None, None, None, :].expand(-1, S, H, W, 3)  # (B, S, H, W, 3)
            return origins, dirs
        else:
            tan_half_fovy = torch.tan(
                torch.as_tensor(self.fov_y_deg, dtype=torch.float32, device=device) * (torch.pi / 180.0) * 0.5
            )
            aspect = torch.as_tensor(self.aspect, dtype=torch.float32, device=device)
            px = xv * tan_half_fovy * aspect  # (S, H, W)
            py = -yv * tan_half_fovy          # (S, H, W)

            forward = self.forward  # (3,)
            right = self.right      # (3,)
            up = self.up            # (3,)

            dirs = (
                px[..., None] * right[None, None, None, :] +
                py[..., None] * up[None, None, None, :] +
                forward[None, None, None, :]
            )  # (S, H, W, 3)
            dirs = torch.nn.functional.normalize(dirs, dim=-1, eps=1e-8)

            origins = self.origin[None, None, None, :].expand(S, H, W, 3)  # (S, H, W, 3)
            return origins, dirs


def _look_at_basis(origin: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forward = torch.nn.functional.normalize(target - origin, dim=0, eps=1e-8)
    right = torch.nn.functional.normalize(torch.cross(forward, up, dim=0), dim=0, eps=1e-8)
    true_up = torch.nn.functional.normalize(torch.cross(right, forward, dim=0), dim=0, eps=1e-8)
    return forward, right, true_up


def _compute_vertical_fov_deg(focal_length_mm: float, sensor_height_mm: float) -> float:
    return float(2.0 * torch.atan(torch.tensor(sensor_height_mm * 0.5) / torch.tensor(focal_length_mm)) * (180.0 / torch.pi))


def create_camera_from_mitsuba(sensor: Dict, device: torch.device | str = "cpu") -> PerspectiveCamera:
    device = torch.device(device)

    # Resolution: allow either (H,W) or list of such; require same across batch if list provided
    res = sensor.get("resolution", (512, 512))
    if isinstance(res, (list, tuple)) and len(res) == 2 and not isinstance(res[0], (list, tuple)):
        res_h, res_w = int(res[0]), int(res[1])
        aspect = float(res_w) / float(res_h)
    else:
        # Assume list of (H,W) with same values; compute once
        first = res[0]
        res_h, res_w = int(first[0]), int(first[1])
        aspect = float(res_w) / float(res_h)

    # Prefer explicit fov if provided; allow batched list/tensor
    fov_val = sensor.get("fov", sensor.get("fov_deg", None))
    if fov_val is not None:
        fov_y_deg = torch.as_tensor(fov_val, dtype=torch.float32, device=device)
    else:
        # Legacy focal length fields; allow scalar only
        focal_length_mm = float(sensor.get("focal_length_mm"))
        sensor_width_mm = float(sensor.get("sensor_width_mm"))
        sensor_height_mm = sensor.get("sensor_height_mm")
        if sensor_height_mm is None:
            sensor_height_mm = sensor_width_mm / aspect
        sensor_height_mm = float(sensor_height_mm)
        fov_y_deg = _compute_vertical_fov_deg(focal_length_mm, sensor_height_mm)

    # Build camera basis from Mitsuba-style to_world (camera -> world). Allow batched.
    if sensor.get("to_world") is not None:
        tw = torch.tensor(sensor["to_world"], dtype=torch.float32, device=device)
        # Squeeze any trailing singleton dims that sometimes appear (e.g., (4,4,1) -> (4,4))
        while tw.dim() >= 2 and tw.shape[-1] == 1 and tw.shape[-2] != 1:
            tw = tw.squeeze(-1)
        # Ensure last two dims are 4x4; if not, try squeezing once more
        if tw.dim() > 2 and tw.shape[-2:] != (4, 4) and tw.shape[-1] == 1:
            tw = tw.squeeze(-1)
        if tw.dim() == 2:
            tw = tw.view(4, 4)
            origin = tw[:3, 3]
            ex = tw[:3, 0]
            ey = tw[:3, 1]
            ez = tw[:3, 2]
            right = torch.nn.functional.normalize(-ex, dim=0, eps=1e-8)
            up = torch.nn.functional.normalize(ey, dim=0, eps=1e-8)
            forward = torch.nn.functional.normalize(ez, dim=0, eps=1e-8)
        else:
            # (B,4,4)
            B = tw.shape[0]
            origin = tw[:, :3, 3]
            ex = tw[:, :3, 0]
            ey = tw[:, :3, 1]
            ez = tw[:, :3, 2]
            right = torch.nn.functional.normalize(-ex, dim=1, eps=1e-8)
            up = torch.nn.functional.normalize(ey, dim=1, eps=1e-8)
            forward = torch.nn.functional.normalize(ez, dim=1, eps=1e-8)
    else:
        # Fallback: allow scalar or batched look-at
        origin_in = sensor.get("origin", [0.0, 0.0, 3.0])
        target_in = sensor.get("target", [0.0, 0.0, 0.0])
        up_in = sensor.get("up", [0.0, 1.0, 0.0])
        origin_t = torch.tensor(origin_in, dtype=torch.float32, device=device)
        target_t = torch.tensor(target_in, dtype=torch.float32, device=device)
        up_t = torch.tensor(up_in, dtype=torch.float32, device=device)
        if origin_t.dim() == 1:
            forward, right, up = _look_at_basis(origin_t, target_t, up_t)
            origin = origin_t
        else:
            # Batched look-at
            forward = torch.nn.functional.normalize(target_t - origin_t, dim=1, eps=1e-8)
            right = torch.nn.functional.normalize(torch.cross(forward, up_t, dim=1), dim=1, eps=1e-8)
            up = torch.nn.functional.normalize(torch.cross(right, forward, dim=1), dim=1, eps=1e-8)
            origin = origin_t

    return PerspectiveCamera(
        origin=origin,
        forward=forward,
        right=right,
        up=up,
        fov_y_deg=float(fov_y_deg.item()) if torch.is_tensor(fov_y_deg) and fov_y_deg.dim() == 0 else fov_y_deg,
        aspect=aspect,
        device=device,
    )


