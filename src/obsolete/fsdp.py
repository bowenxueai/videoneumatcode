# fsdp_min_manual_wrap_fixed.py
# Run: torchrun --nproc_per_node=4 fsdp_min_manual_wrap_fixed.py

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import enable_wrap, wrap


class Model(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.layer1 = nn.Linear(10,20)
        self.layer2 = nn.Linear(20,30)

# ---------------- Main ----------------
if __name__ == "__main__":
    ###############################
    # Init distributed
    ###############################
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    ###############################
    # Create model *on device*
    ###############################
    model = Model().to(device)

    print(f"\n===== RANK {rank} (GPU {local_rank}) =====")
    for name, p in model.named_parameters():
            print(
                f"unit '{name}': "
                f"local shard numel = {p.numel():,}, "
                f"device = {p.device}"
            )

    ###############################
    # Manual wrapping
    ###############################
    with enable_wrap(wrapper_cls=FSDP, device_id=device):
        model.layer1 = wrap(model.layer1)
        model.layer2 = wrap(model.layer2)
        model = wrap(model)  # root FSDP

    print(f"\n===== RANK {rank} (GPU {local_rank}) =====")
    for name, p in model.named_parameters():
            print(
                f"FSDP unit '{name}': "
                f"local shard numel = {p.numel():,}, "
                f"device = {p.device}"
            )

    dist.destroy_process_group()
