accelerate launch src/inverse_render.py \
 train_dataset=[matsynth8_train_sparse_10,matsynth8_train_dense_10] \
 val_dataset=[matsynth8_train_sparse_10,matsynth8_train_dense_10] \
 resolution=256 \
 batch_size=8 \
 ${@}