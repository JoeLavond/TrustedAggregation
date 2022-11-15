screen -dmS c10-nt-trust-one-alpha bash -c "
    source activate joeenv;
    python trust.py --alpha=1 --n_rounds=250 --n_malicious=1 --gpu_start=0;
"

screen -dmS c10-nt-trust-two-alpha bash -c "
    source activate joeenv;
    python trust.py --alpha=1 --n_rounds=250 --dba=1 --n_malicious=2 --col_size=2 --gpu_start=0;
"

screen -dmS c10-nt-trust-four-alpha bash -c "
    source activate joeenv;
    python trust.py --alpha=1 --n_rounds=250 --dba=1 --n_malicious=4 --col_size=2 --row_size=2 --gpu_start=0;
"

