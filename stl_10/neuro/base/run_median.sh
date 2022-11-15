# screen -dmS re-s10-nt-median-one bash -c "
    # source activate joeenv;
    # CUDA_VISIBLE_DEVICES=0,2 python robust.py --n_rounds=250 --trim_mean=0 --n_malicious=1 --gpu_start=0;
# "
#
# screen -dmS re-s10-nt-median-two bash -c "
    # source activate joeenv;
    # python robust.py --n_rounds=250 --trim_mean=0 --dba=1 --n_malicious=2 --col_size=12 --gpu_start=0;
# "
#
screen -dmS re-s10-nt-median-four bash -c "
    source activate joeenv;
    python robust.py --n_rounds=250 --trim_mean=0 --dba=1 --n_malicious=4 --col_size=12 --row_size=12 --gpu_start=0;
"
