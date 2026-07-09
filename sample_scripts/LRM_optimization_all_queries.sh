accelerate launch src/inverse_render_LRM.py batch_size=4 \
    resolution=256 \
    train_dataset=[matsynth8q_train_sparse_all] \
    val_dataset=[matsynth8q_train_sparse_10,matsynth8q_train_dense_10,matsynth8q_train_sparse_hardest10,matsynth8q_train_dense_hardest10,diffusion_test_sparse_all] \
    checkpoint.network=/lustre/fsw/portfolios/nvr/users/shadadan/neugen/outputs/slurm/26039679/standalone-res256-matsynth8_train_sparse_512-offset_True-TM_srgb_no_clamp-net_l12-64x4-bs8/mlp.pth \
    n_iterations=5000 \
    loss_func=l2log \
    use_hdf5=true \
    ${@}