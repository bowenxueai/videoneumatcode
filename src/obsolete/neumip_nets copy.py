import torch
import torch.nn as nn
import json

    


class BRDFNetwrok(nn.Module):
    def __init__(self, desc) -> None:
        super().__init__()
        # self.desc = desc
        # cfg = desc['model']['config']
        # layers = desc['model']['layers']

        # self.layers = []

        # active_fn = lambda name: getattr(nn, name)(inplace=False)

        
        # for i in range(len(layers)):
        #     inp, out = layers[i]["n_neurons_previous"], layers[i]["n_neurons"]
        #     nn_layer = nn.Linear(inp, out, bias = cfg['bias'])
        #     with torch.no_grad():
        #         nn_layer.weight.data = torch.tensor(layers[i]["weights"], dtype=torch.float32).reshape(nn_layer.weight.data.shape)
        #         if cfg['bias']:
        #             nn_layer.bias.data = torch.tensor(layers[i]["biases"], dtype=torch.float32).reshape(nn_layer.bias.data.shape)

        #     layer = [nn_layer]
        #     is_last_layer = i == len(layers) -1
        #     if  not is_last_layer:
        #         layer += [active_fn(cfg['activation_fn'])]
        #     else:
        #         last_activation = cfg["output_activation_fn"]
                
        #         if last_activation is not None and last_activation.lower()!='none':
        #             layer += [active_fn(cfg["output_activation_fn"])]

        #     self.layers += layer
        self.latents = torch.nn.Parameter(torch.randn(1, 7, 512, 512))            
        self.network = nn.Sequential(nn.Linear(11, 25), nn.ReLU(), nn.Linear(25, 25), nn.ReLU(), nn.Linear(25, 25), nn.ReLU(), nn.Linear(25, 3), nn.ReLU())
    
    def forward(self, wi, wo, latents):
        # latents = torch.nn.functional.grid_sample(self.latents, 2*uv[None, :,None,:] - 1, mode = 'bilinear', align_corners = False)
        # latents = latents.squeeze(0).squeeze(-1).transpose(0,1)
        # # print(latents.shape, wo.shape, wi.shape)
        # # print(uv.min(dim = 0).values, uv.max(dim = 0).values)
        # print(latents.shape, wo.shape, wi.shape)
        x = torch.cat([wi, wo, latents], dim = -1)
        # print('wi', wi.shape, torch.min(wi, dim = 0).values, torch.max(wi, dim = 0).values)
        # print('wo', wo.shape, torch.min(wo, dim = 0).values, torch.max(wo, dim = 0).values)
        # print('latents',latents.shape, latents.min(), latents.max())
#        print('out', y.shape, torch.min(y).values, torch.max(y).values)

        return self.network(x)


def load_network(path, device):
    if not path:
        return BRDFNetwrok(None).to(device)

    f = open (path, "r")
    desc = json.loads(f.read())
    network = BRDFNetwrok(desc).to(device)    
    
    return network


