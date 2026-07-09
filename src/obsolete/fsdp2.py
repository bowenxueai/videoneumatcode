# fsdp_params.py
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import enable_wrap, wrap

class ParamModule(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.param = nn.Parameter(torch.randn(*shape))
    def forward(self, x):
        return None

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ParamModule((10,20))
        self.layer2 = ParamModule((10,30))
        
    def forward(self, x):
        return None
if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    model = Model().to(device)

    print(f"\n===== BEFORE WRAP rank{rank} =====")
    for name, p in model.named_parameters():
        print(name, p.numel(), p.device)

    # ----------- MANUAL WRAP ------------
    with enable_wrap(wrapper_cls=FSDP, device_id=device):
        model.layer1 = wrap(model.layer1)
        model.layer2 = wrap(model.layer2)
        model = wrap(model)  # root FSDP

    # Materialize lazy flattening
    model(torch.randn(1, device=device))

    print(f"\n===== AFTER WRAP rank{rank} =====")
    for name, p in model.named_parameters():
        print(
            f"{name}: local shard = {p.numel()} / device = {p.device}"
        )

    dist.destroy_process_group()
