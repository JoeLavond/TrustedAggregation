screen -dmS alpha-nt-four bash -c "
    source activate joeenv;
    python train.py --n_rounds=250 --alpha=1 --alpha_val=1 --d_scale=1.5 --dba=1 --n_malicious=4 --col_size=2 --row_size=2 --gpu_start=2;
"

