screen -dmS alpha-one bash -c "
    source activate joeenv;
    python train.py --n_rounds=250 --alpha=1 --alpha_val=1 --d_scale=2 --n_malicious=1 --gpu_start=0;
"

screen -dmS alpha-two bash -c "
    source activate joeenv;
    python train.py --n_rounds=250 --alpha=1 --alpha_val=1 --d_scale=2 --dba=1 --n_malicious=2 --col_size=2 --gpu_start=0;
"

screen -dmS alpha-four bash -c "
    source activate joeenv;
    python train.py --n_rounds=250 --alpha=1 --alpha_val=1 --d_scale=2 --dba=1 --n_malicious=4 --col_size=2 --row_size=2 --gpu_start=0;
"

