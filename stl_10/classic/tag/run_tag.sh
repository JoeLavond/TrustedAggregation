# screen -dmS re-s10-tag-one bash -c "
    # source activate tag;
    # python train.py --d_scale=1.1 --n_rounds=250 --n_malicious=1 --gpu_start=0;
# "

screen -dmS re-s10-tag-two bash -c "
    source activate tag;
    python train.py --d_scale=1.1 --n_rounds=250 --dba=1 --n_malicious=2 --col_size=12 --gpu_start=2;
"

# screen -dmS re-s10-tag-four bash -c "
    # source activate tag;
    # python train.py --d_scale=1.1 --n_rounds=250 --dba=1 --n_malicious=4 --col_size=12 --row_size=12 --gpu_start=0;
# "

