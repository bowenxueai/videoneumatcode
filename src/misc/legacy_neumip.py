import torch
import torch.nn as nn
import json




class BRDFNetwrok(nn.Module):
    # the torch model for NeuMIP
    def __init__(self, use_offset = True, resolution = 1024, latent_channels = 7, uv_scale=1, network_width=25, network_depth=3, uv_shift = (0.0, 0.0)) -> None:
        super().__init__()
        self.use_offset = use_offset
        self.uv_scale = uv_scale
        self.return_offset = False
        self.uv_shift = uv_shift
        self.latents = None            
        self.mlp = nn.Sequential(
                *[nn.Linear(latent_channels + 4, network_width), 
                *[l for _ in range(network_depth) for l in (nn.LeakyReLU(inplace=True), nn.Linear(network_width, network_width))],
                nn.LeakyReLU(inplace=True),
                nn.Linear(network_width, 3), 
                nn.LeakyReLU(inplace=True)]
            )

        if self.use_offset:
            self.offset_latents = None           
            self.offset_mlp = nn.Sequential(
                    *[nn.Linear(latent_channels + 2, network_width), 
                    *[l for _ in range(network_depth) for l in (nn.LeakyReLU(inplace=True), nn.Linear(network_width, network_width))],
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(network_width, 1)]
                )

    def my_grid_sample(self, input, grid):
        uv_translate = self.uv_shift
        # Step 1: Apply Mitsuba's transform: (uv + translate) * scale
        mitsuba_x = (grid[:, 0] + uv_translate[0]) * self.uv_scale
        mitsuba_y = (grid[:, 1] + uv_translate[1]) * self.uv_scale
        
        # Step 2: Convert to grid_sample [-1, 1] space
        # X: standard conversion
        grid_x = 2 * mitsuba_x - 1
        # Y: flip to compensate for bitmap NOT being flipped (Mitsuba flips bitmap, we don't)
        grid_y = 1 - 2 * mitsuba_y  # = -(2 * mitsuba_y - 1)
        
        grid = torch.stack([grid_x, grid_y], dim=-1)[None, :, None, :]
        return torch.nn.functional.grid_sample(input, grid=grid, mode='bilinear', align_corners=False, padding_mode='reflection')



    
    def forward(self, wi, wo, uv):
        # NOTE: Important assumption: wo is the camera direction at first bounce. In mitsuba, wo is the sampled direction.


        z = wo[:,2:3]
        wo = wo[:,0:2]
        wi = wi[:,0:2]

        if self.use_offset:
            offset_latents = self.my_grid_sample(self.offset_latents, uv)
            offset_latents = offset_latents.squeeze(0).squeeze(-1).transpose(0,1)
            x = torch.cat([wo, offset_latents], dim = -1)
            r = self.offset_mlp(x)
            z = torch.clamp(z, min = 0.6)
            shift = r / z*(wo)
            if self.uv_scale != 1:
                shift /= self.uv_scale
            uv1 = uv +  shift
            if self.return_offset:
                return torch.cat([shift, torch.zeros_like(z)], dim = -1)

        if self.use_offset:
            latents = self.my_grid_sample(self.latents, uv1)

        else:
            latents = self.my_grid_sample(self.latents, uv)

        latents = latents.squeeze(0).squeeze(-1).transpose(0,1)

        x = torch.cat([wi, wo, latents], dim = -1)

        return self.mlp(x)


def load_network(desc, device):
    if isinstance(desc, dict):
        return BRDFNetwrok(**desc).to(device)

    f = open (desc, "r")
    desc = json.loads(f.read())
    network = BRDFNetwrok(**desc).to(device)    
    
    return network




