accelerate launch src/inverse_render_LRM_ds.py batch_size=4 \
    resolution=256 \
    train_dataset=[matsynth7_train_sparse_all] \
    val_dataset=[matsynth8_train_sparse_10,matsynth8_train_dense_10,matsynth8_train_sparse_hardest10,matsynth8_train_dense_hardest10,diffusion_test_sparse_all] \
    checkpoint.network=/lustre/fsw/portfolios/nvr/users/shadadan/neugen/outputs/good/standalone-matsynth512-256-offset_True-TM_srgb_no_clamp-net_l12-64x4-bs32/model.pth \
    n_iterations=10000 \
    ${@}

# dataset.frames_info.every_other: how many frames to train on, set 8 if you want to load and train on only every 8 frames of the dataset: range(0, 129, step=8)