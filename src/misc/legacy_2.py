import torch
import torch.nn as nn
import json




class BRDFNetwrok(nn.Module):
    # the torch model for NeuMIP
    def __init__(self, use_offset = True, resolution = 1024, latent_channels = 7, uv_scale=1, network_width=25, network_depth=3) -> None:
        super().__init__()
        self.use_offset = use_offset
        self.uv_scale = uv_scale
        self.return_offset = False
        self.return_offset_as_separate_output = False
        self.latents = torch.nn.Parameter(torch.randn(1, latent_channels,resolution, resolution)*0.1)            
        self.network = nn.Sequential(
                *[nn.Linear(latent_channels + 4, network_width), 
                *[l for _ in range(network_depth) for l in (nn.LeakyReLU(inplace=True), nn.Linear(network_width, network_width))],
                nn.LeakyReLU(inplace=True),
                nn.Linear(network_width, 3), 
                nn.LeakyReLU(inplace=True)]
            )

        if self.use_offset:
            self.offset_latents = torch.nn.Parameter(torch.randn(1, latent_channels,resolution, resolution)*0.1)            
            self.offset_network = nn.Sequential(
                    *[nn.Linear(latent_channels + 2, network_width), 
                    *[l for _ in range(network_depth) for l in (nn.LeakyReLU(inplace=True), nn.Linear(network_width, network_width))],
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(network_width, 1)]
                )

    def my_grid_sample(self, input, grid):
        new_grid = torch.zeros_like(grid)
        new_grid[:, 0] = grid[:, 0] 
        new_grid[:, 1] = 1 - grid[:, 1]          # flip the y coordinate
        return torch.nn.functional.grid_sample(input, grid = 2*new_grid[None, :,None,:] - 1, mode = 'bilinear', align_corners = False)



    
    def forward(self, wi, wo, uv):
        # NOTE: Important assumption: wo is the camera direction at first bounce. In mitsuba, wo is the sampled direction.


        flattened = False
        if wi.ndim > 2:
            flattened = True
            s,h,w,c = wi.shape
            wi = wi.view(-1, 3)
            wo = wo.view(-1, 3)
            uv = uv.view(-1, 2)

        offset_out = torch.zeros_like(wo)
        if self.use_LRM:
        if self.uv_scale != 1:
            uv *= self.uv_scale

        z = wo[:,2:3]
        wo = wo[:,0:2]
        wi = wi[:,0:2]

        if self.use_offset:
            offset_latents = self.my_grid_sample(self.offset_latents, uv)
            offset_latents = offset_latents.squeeze(0).squeeze(-1).transpose(0,1)
            x = torch.cat([wo, offset_latents], dim = -1)
            r = self.offset_network(x)
            z = torch.clamp(z, min = 0.6)
            shift = r / z*(wo)
            if self.uv_scale != 1:
                shift *= self.uv_scale
            uv1 = uv +  shift
            if self.return_offset or self.return_offset_as_separate_output:
                offset_out =  torch.cat([shift, torch.zeros_like(z)], dim = -1)
                if flattened:
                    offset_out = offset_out.view(s,h,w,3)
                if self.return_offset:
                    return offset_out

        if self.use_offset:
            latents = self.my_grid_sample(self.latents, uv1)

        else:
            latents = self.my_grid_sample(self.latents, uv)

        latents = latents.squeeze(0).squeeze(-1).transpose(0,1)

        x = torch.cat([wi, wo, latents], dim = -1)

        out =  self.network(x)
        if flattened:
            out = out.view(s,h,w,3)

        if self.return_offset_as_separate_output: # this if statement is here to be backward compatible with our mitsuba code.
            return out, offset_out
        return out


def load_network(desc, device):
    if isinstance(desc, dict):
        return BRDFNetwrok(**desc).to(device)

    f = open (desc, "r")
    desc = json.loads(f.read())
    network = BRDFNetwrok(**desc).to(device)    
    
    return network
