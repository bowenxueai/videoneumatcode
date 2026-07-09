import torch
import torch.nn as nn
import json

class SmeLU(nn.Module):
    def __init__(self, inplace = False):
        super(SmeLU, self).__init__()
        self.alpha =  0.5
        self.beta = 0.01
        self.lo = -self.alpha
        self.hi = self.alpha
        self.A = (1.0 - self.beta) / (4.0 * self.alpha)
        self.B = (1.0 + self.beta) / 2.0
        self.C = self.alpha * (1.0 - self.beta) / 4.0
        self.inplace = inplace

    def forward(self, x):
        out = x if self.inplace else x.clone()
        lower_lo = x < self.lo 
        between_lo_hi = (x >= self.lo) & (x < self.hi)
        out[lower_lo] = self.beta * x[lower_lo]
        out[between_lo_hi] = (self.A * x * x + self.B * x + self.C)[between_lo_hi]
        return out
    


class BRDFNetwrok(nn.Module):
    def __init__(self, desc) -> None:
        super().__init__()
        self.desc = desc
        cfg = desc['model']['config']
        layers = desc['model']['layers']

        self.layers = []

        active_fn = lambda name: getattr(nn, name)(inplace=False) if name.lower()!='smelu' else SmeLU(inplace=False)

        for i in range(len(layers)):
            inp, out = layers[i]["n_neurons_previous"], layers[i]["n_neurons"]
            nn_layer = nn.Linear(inp, out, bias = cfg['bias'])
            with torch.no_grad():
                nn_layer.weight.data = torch.tensor(layers[i]["weights"], dtype=torch.float32).reshape(nn_layer.weight.data.shape)
                if cfg['bias']:
                    nn_layer.bias.data = torch.tensor(layers[i]["biases"], dtype=torch.float32).reshape(nn_layer.bias.data.shape)

            layer = [nn_layer]
            is_last_layer = i == len(layers) -1
            if  not is_last_layer:
                layer += [active_fn(cfg['activation_fn'])]
            else:
                last_activation = cfg["output_activation_fn"]
                
                if last_activation is not None and last_activation.lower()!='none':
                    layer += [active_fn(cfg["output_activation_fn"])]

            self.layers += layer
            
        self.network = nn.Sequential(*self.layers)
    
    def forward(self, wi, wo, latents):
        x = torch.cat([wi, wo, latents], dim = -1)
        return torch.exp(self.network(x))


def load_network(path, device):
    f = open (path, "r")
    desc = json.loads(f.read())
    network = BRDFNetwrok(desc).to(device)    
    return network