screen -dmS re-s10-nt-mean-one bash -c "
    source activate tag;
    python robust.py --n_rounds=250 --trim_mean=1 --n_malicious=1 --gpu_start=2;
"

screen -dmS re-s10-nt-mean-two bash -c "
    source activate tag;
    python robust.py --n_rounds=250 --trim_mean=1 --dba=1 --n_malicious=2 --col_size=12 --gpu_start=2;
"

screen -dmS re-s10-nt-mean-four bash -c "
    source activate tag;
    python robust.py --n_rounds=250 --trim_mean=1 --dba=1 --n_malicious=4 --col_size=12 --row_size=12 --gpu_start=2;
"
