#!/bin/sh

srun -A nvr_rtr_rgbxnext \
     --container-image $DISK/cache/neugen.sqsh \
     --container-mounts $DISK/neugen:/home/NeuGen,/lustre:/lustre,/lustre/fsw/portfolios/nvr/users/shadadan/nvoptix.bin:/usr/share/nvidia/nvoptix.bin \
     --pty \
     --nodes 1 \
     --gpus 1 \
     --time 3:59:00 \
     --partition interactive \
     bash

# --container-save $DISK/cache/neugen.sqsh \
#     --container-save $DISK/cache/neugen_neuapp.sqsh \
     # --container-image /lustre/fsw/portfolios/adlr/projects/adlr_graphics_dlss/users/jlunden/enroot-cache/adlr+generative-rendering+hnet_25.09+2025-11-03.sqsh \
#     --container-image /lustre/fsw/portfolios/adlr/projects/adlr_graphics_dlss/users/jlunden/enroot-cache/adlr+generative-rendering+hnet_25.09+2025-11-03.sqsh \
#     --container-image /lustre/fsw/portfolios/nvr/projects/nvr_rtr_rgbxnext/users/zhzeng/containers/wan2.1:2025-08-01.sqsh \
