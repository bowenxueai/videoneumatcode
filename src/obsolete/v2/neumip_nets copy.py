import torch
import torch.nn as nn
import json
from einops import rearrange


class BRDFNetwrok(nn.Module):
    # the torch model for NeuMIP
    def __init__(self, use_offset = True, resolution = 1024, latent_channels = 7, uv_scale=1, network_width=25, network_depth=3, standalone_latents = True, num_materials = 1) -> None:
        super().__init__()
        self.use_offset = use_offset
        self.uv_scale = uv_scale
        self.return_offset = False
        self.return_offset_as_separate_output = False
        self.num_materials = num_materials
        self.latents = torch.nn.Parameter(torch.randn(num_materials, latent_channels,resolution, resolution)*0.1) if standalone_latents else None          
        self.network = nn.Sequential(
                *[nn.Linear(latent_channels + 4, network_width), 
                *[l for _ in range(network_depth) for l in (nn.LeakyReLU(inplace=True), nn.Linear(network_width, network_width))],
                nn.LeakyReLU(inplace=True),
                nn.Linear(network_width, 3), 
                nn.LeakyReLU(inplace=True)]
            )

        if self.use_offset:
            self.offset_latents = torch.nn.Parameter(torch.randn(num_materials,latent_channels, resolution, resolution)*0.1)  if standalone_latents else None          
            self.offset_network = nn.Sequential(
                    *[nn.Linear(latent_channels + 2, network_width), 
                    *[l for _ in range(network_depth) for l in (nn.LeakyReLU(inplace=True), nn.Linear(network_width, network_width))],
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(network_width, 1)]
                )

        self.material_id = 0


    def my_grid_sample(self, input, grid):
        new_grid = torch.zeros_like(grid)
        new_grid[..., 0] = grid[..., 0] 
        new_grid[..., 1] = 1 - grid[..., 1]          # flip the y coordinate
        print('uv lookup before flatten', new_grid.shape)
        flattened = False
        if new_grid.ndim > 4:
            flattened = True
            og_shape = new_grid.shape[0:-3]
            new_grid = new_grid.flatten(start_dim = 0, end_dim = -4)
        B = new_grid.shape[0]
        print('uv lookup after flatten', new_grid.shape)
        print('grid sample texture', input.shape)

        out =  torch.nn.functional.grid_sample(input.expand(B, *([-1] * (input.ndim - 1))), grid = 2*new_grid - 1, mode = 'bilinear', align_corners = False)
        print('grid sample out', out.shape)

        if flattened:
            print('unflattening', og_shape)
            out = out.unflatten(0, og_shape)
        print('grid sample out', out.shape)

        out = rearrange(out, '... c h w -> ... h w c')
        return out



    
    def forward(self, wi, wo, uv):
        # NOTE: Important assumption: wo is the camera direction at first bounce. In mitsuba, wo is the sampled direction.

        offset_out = torch.zeros_like(wo)

        print('uv', uv.shape)

        flattened = False
        # if wi.ndim > 2:
        #     flattened = True
        #     og_shape = wi.shape[:-1]
        #     wi = wi.view(-1, 3)
        #     wo = wo.view(-1, 3)
        #     uv = uv.view(-1, 2)

        if self.uv_scale != 1:
            uv *= self.uv_scale

        z = wo[...,2:3]
        wo = wo[...,0:2]
        wi = wi[...,0:2]

        if self.use_offset:
            offset_latents = self.my_grid_sample(self.offset_latents[self.material_id], uv)
            print('offset latents', offset_latents.shape, 'wo', wo.shape, 'wi', wi.shape, 'uv', uv.shape)
            x = torch.cat([wo, offset_latents], dim = -1)
            r = self.offset_network(x)
            z = torch.clamp(z, min = 0.6)
            shift = r / z*(wo)
            if self.uv_scale != 1:
                shift *= self.uv_scale
            uv1 = uv +  shift
            if self.return_offset or self.return_offset_as_separate_output:
                offset_out =  torch.cat([shift, torch.zeros_like(z)], dim = -1)
                # if flattened:
                #     offset_out = offset_out.view(*og_shape,3)
                if self.return_offset:
                    return offset_out

        if self.use_offset:
            latents = self.my_grid_sample(self.latents[self.material_id], uv1)

        else:
            latents = self.my_grid_sample(self.latents[self.material_id], uv)

        # latents = latents.squeeze(0).squeeze(-1).transpose(0,1)

        x = torch.cat([wi, wo, latents], dim = -1)

        out =  self.network(x)
        # if flattened:
        #     out = out.view(*og_shape,3)

        if self.return_offset_as_separate_output: # this if statement is here to be backward compatible with our mitsuba code.
            # offset_out = offset_out.view(*og_shape,3)
            return out, offset_out
        return out


def load_network(desc, device):
    if isinstance(desc, dict):
        return BRDFNetwrok(**desc).to(device)

    f = open (desc, "r")
    desc = json.loads(f.read())
    network = BRDFNetwrok(**desc).to(device)    
    
    return network



