accelerate launch src/inverse_render_LRM.py \
    train_dataset=['diffusion_i2v_sparse_all'] \
    val_dataset=['diffusion_1024x1024_sparse_all'] \
    optimize=False \
    date=inference \
    checkpoint.latent='/lustre/fs12/portfolios/nvr/projects/nvr_rtr_neugen/users/shadadan/neugen/outputs/good/dense/23268740_longrun_dense_final_1024/wan_lrm-res1024-matsynth8_train_sparse_all-matsynth8_train_dense_all-offset_True-TM_srgb_no_clamp-net_l12-64x4-bs1/latent_module.pth' \
    checkpoint.network='/lustre/fs12/portfolios/nvr/projects/nvr_rtr_neugen/users/shadadan/neugen/outputs/good/dense/23268740_longrun_dense_final_1024/wan_lrm-res1024-matsynth8_train_sparse_all-matsynth8_train_dense_all-offset_True-TM_srgb_no_clamp-net_l12-64x4-bs1/mlp.pth' \
    latent/upsampler=vae \
    network.min_z=0.64 \
    time='final' \
    network.max_offset=0.02 \
    resolution=1024 \
    ${@}

    # time='final' \
    # checkpoint.latent='outputs/good/dense/23268740_longrun_dense_final_1024/wan_lrm-res1024-matsynth8_train_sparse_all-matsynth8_train_dense_all-offset_True-TM_srgb_no_clamp-net_l12-64x4-bs1/latent_module.pth' \
    # checkpoint.network='outputs/good/dense/23268740_longrun_dense_final_1024/wan_lrm-res1024-matsynth8_train_sparse_all-matsynth8_train_dense_all-offset_True-TM_srgb_no_clamp-net_l12-64x4-bs1/mlp.pth' \
