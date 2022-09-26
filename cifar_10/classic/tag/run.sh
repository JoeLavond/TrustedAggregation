source activate joeenv;

python train.py --n_rounds=50 --alpha=10000 --alpha_val=10000 --d_start=1 --m_start=1 --dba=0 --n_malicious=1 --n_val_data=100;
python train.py --n_rounds=50 --alpha=10000 --alpha_val=10000 --d_start=1 --m_start=1 --dba=1 --n_malicious=2 --col_size=2 --n_val_data=100;
python train.py --n_rounds=50 --alpha=10000 --alpha_val=10000 --d_start=1 --m_start=1 --dba=1 --n_malicious=4 --col_size=2 --row_size=2 --n_val_data=100;

