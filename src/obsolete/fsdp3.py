# fsdp_params_accelerate.py
import os
import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from accelerate.utils import FullyShardedDataParallelPlugin

# =======================
#  MODULE DEFINITIONS
# =======================
class ParamModule(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.param = nn.Parameter(torch.randn(*shape))

    def forward(self, x):
        return None


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ParamModule((10, 20))
        self.layer2 = ParamModule((10, 30))

    def forward(self, x):
        return None


# =======================
#  MAIN
# =======================
if __name__ == "__main__":
    fsdp_plugin = FullyShardedDataParallelPlugin(
        sharding_strategy="FULL_SHARD",
    )

    accelerator = Accelerator(
        mixed_precision="no",
        fsdp_plugin=fsdp_plugin
    )

    rank = accelerator.process_index
    device = accelerator.device

    # -----------------------------
    # Build model on device
    # -----------------------------
    model = Model().to(device)

    print(f"\n===== BEFORE WRAP rank{rank} =====")
    for name, p in model.named_parameters():
        print(name, p.shape, p.numel(), p.device)

    # # -----------------------------
    # # Manual wrap (same as pure FSDP)
    # # -----------------------------
    with enable_wrap(wrapper_cls=FSDP, device_id=device):
        wrap(model.layer1)
        wrap(model.layer2)
        wrap(model)    # root
    print(isinstance(model, FSDP))
    # -----------------------------
    # Materialize lazy flatten
    # -----------------------------
    # model(torch.randn(1, device=device))

    # -----------------------------
    # Prepare with accelerate
    # -----------------------------
    model = accelerator.prepare(model)

    # -----------------------------
    # Inspect shards
    # -----------------------------
    accelerator.wait_for_everyone()
    print(f"\n===== AFTER WRAP rank{rank} =====")
    for name, p in model.named_parameters():
        print(
            f"{name}: shard shape={tuple(p.shape)}, "
            f"numel={p.numel()}, device={p.device}"
        )
