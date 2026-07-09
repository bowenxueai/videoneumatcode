accelerate launch src/inverse_render.py \
 train_dataset=[matsynth8_train_sparse_512,matsynth8_train_dense_512] \
 val_dataset=[matsynth8_train_sparse_512first10,matsynth8_train_dense_512first10] \
 resolution=256 \
 batch_size=32 \
 n_iterations=50000 \
 network.min_z=0.4 \
 ${@}