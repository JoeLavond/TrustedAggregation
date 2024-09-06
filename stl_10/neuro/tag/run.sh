source activate tag;
CUDA_VISIBLE_DEVICES=3 python train.py --n_rounds=250 --n_malicious=1 --col_size=12 --row_size=12 --d_scale=2;
