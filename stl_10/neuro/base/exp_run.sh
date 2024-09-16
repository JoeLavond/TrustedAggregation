screen -dmS s10-nt-trust-four bash -c "
    source activate tag;
    python trust.py --n_rounds=250 --dba=1 --n_malicious=4 --col_size=12 --row_size=12 --gpu_start=0;
"
