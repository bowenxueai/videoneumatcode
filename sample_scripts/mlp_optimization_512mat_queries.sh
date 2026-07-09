accelerate launch src/inverse_render.py \
 train_dataset=[matsynth8_train_sparse_512] \
 val_dataset=[matsynth8_train_sparse_512first10,matsynth8_train_dense_512first10] \
 resolution=256 \
 batch_size=16 \
 n_iterations=15000 \
 use_hdf5=true \
 loss_func=l2log \
 ${@}