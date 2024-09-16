screen -dmS c10-trust-one-alpha bash -c "
    source activate tag;
    python trust.py --alpha=1 --n_rounds=250 --n_malicious=1 --gpu_start=2;
"

screen -dmS c10-trust-two-alpha bash -c "
    source activate tag;
    python trust.py --alpha=1 --n_rounds=250 --dba=1 --n_malicious=2 --col_size=2 --gpu_start=2;
"

screen -dmS c10-trust-four-alpha bash -c "
    source activate tag;
    python trust.py --alpha=1 --n_rounds=250 --dba=1 --n_malicious=4 --col_size=2 --row_size=2 --gpu_start=2;
"

