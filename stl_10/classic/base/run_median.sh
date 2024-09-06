screen -dmS re-s10-median-two bash -c "
    source activate tag;
    python robust.py --n_rounds=250 --trim_mean=0 --dba=1 --n_malicious=2 --col_size=12 --gpu_start=0;
"

screen -dmS re-s10-median-four bash -c "
    source activate tag;
    python robust.py --n_rounds=250 --trim_mean=0 --dba=1 --n_malicious=4 --col_size=12 --row_size=12 --gpu_start=0;
"
