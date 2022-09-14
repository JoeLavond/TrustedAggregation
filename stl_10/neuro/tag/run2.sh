source activate joeenv;

CUDA_VISIBLE_DEVICES=2,3 python train.py --n_rounds=100 --alpha=10000 --alpha_val=10000 --d_start=1 --m_start=1 --dba=0 --n_malicious=1;
CUDA_VISIBLE_DEVICES=2,3 python train.py --n_rounds=100 --alpha=10000 --alpha_val=10000 --d_start=101 --m_start=1 --dba=0 --n_malicious=1;

CUDA_VISIBLE_DEVICES=2,3 python train.py --n_rounds=100 --alpha=10000 --alpha_val=10000 --d_start=1 --m_start=1 --dba=1 --n_malicious=2 --col_size=12;
CUDA_VISIBLE_DEVICES=2,3 python train.py --n_rounds=100 --alpha=10000 --alpha_val=10000 --d_start=101 --m_start=1 --dba=1 --n_malicious=2 --col_size=12;

CUDA_VISIBLE_DEVICES=2,3 python train.py --n_rounds=100 --alpha=10000 --alpha_val=10000 --d_start=1 --m_start=1 --dba=1 --n_malicious=4 --col_size=12 --row_size=12;
CUDA_VISIBLE_DEVICES=2,3 python train.py --n_rounds=100 --alpha=10000 --alpha_val=10000 --d_start=101 --m_start=1 --dba=1 --n_malicious=4 --col_size=12 --row_size=12;
