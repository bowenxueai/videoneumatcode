accelerate launch src/inverse_render_LRM.py batch_size=4 \
    resolution=256 \
    checkpoint.network=outputs/good/standalone-matsynth10-256-offset_True-TM_srgb_no_clamp-net_l12-64x4-bs16/model.pth \
    n_iterations=20000 \
    ${@}

# dataset.frames_info.every_other: how many frames to train on, set 8 if you want to load and train on only every 8 frames of the dataset: range(0, 129, step=8)