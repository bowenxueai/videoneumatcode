accelerate launch src/inverse_render.py \
    train_dataset=[matsynth8_train_sparse_leather] \
    val_dataset=[matsynth8_train_sparse_leather] \
    resolution=1024 \
    batch_size=4 \
    network=medium \
    ${@}