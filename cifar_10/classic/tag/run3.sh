source activate joeenv;

CUDA_VISIBLE_DEVICES=2,3 python train.py --n_rounds=50 --alpha=1 --alpha_val=10000 --d_start=1 --m_start=1 --dba=1 --n_malicious=4 --col_size=2 --row_size=2 --d_scale=1.4;
