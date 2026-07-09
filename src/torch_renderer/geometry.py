from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
import imageio.v3 as iio
import numpy as np



@dataclass
class Plane:
    width: float
    height: float
    plane_point: torch.Tensor  # (3,)
    plane_normal: torch.Tensor  # (3,), normalized
    device: torch.device

    @staticmethod
    def create(width: float = 2.0, height: float = 2.0, device: torch.device | str = "cpu") -> "TexturedPlane":
        device = torch.device(device)
        return Plane(
            width=width,
            height=height,
            plane_point=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device),
            plane_normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device),
            device=device,
        )

    def size(self) -> tuple[float, float]:
        """Return the physical plane size as (width, height)."""
        return float(self.width), float(self.height)

