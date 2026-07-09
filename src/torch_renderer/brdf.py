import torch
from .texture import Texture


class DiffuseBRDF(torch.nn.Module):
    """Lambertian BRDF: f_r = albedo / pi.

    The integrator is expected to include the cosine and distance falloff terms.
    """

    def __init__(self, reflectance: str | list[float], device: torch.device | str = "cpu", srgb: bool = True):
        super().__init__()
        if isinstance(reflectance, str):
            self.albedo = Texture.from_file(reflectance, device=device, srgb=True)
        else:   
            self.albedo = torch.tensor(reflectance, dtype=torch.float32, device=device)


    def forward(self, wi: torch.Tensor, wo: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        # Albedo via grid_sample: flatten arbitrary leading dims to S for lookup
        if isinstance(self.albedo, Texture):
            if uv.dim() <= 4:
                albedo = self.albedo.grid_sample(uv)
            else:
                H, W = uv.shape[-3], uv.shape[-2]
                leading = uv.shape[:-3]
                S = 1
                for d in leading:
                    S *= int(d)
                uv_flat = uv.reshape(S, H, W, 2)
                albedo_flat = self.albedo.grid_sample(uv_flat)  # (S,H,W,3)
                albedo = albedo_flat.reshape(*leading, H, W, 3)
        else:
            # Constant RGB broadcast to match uv's leading dims and spatial size
            target_shape = (*uv.shape[:-1], 3)
            base = self.albedo.view(*([1] * (uv.dim() - 1)), 3)
            albedo = base.expand(*target_shape)

        return (albedo) / torch.pi


