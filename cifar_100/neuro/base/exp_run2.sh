screen -dmS c100-nt-trust-one bash -c "
    source activate tag;
    python trust2.py --n_rounds=250 --n_malicious=1 --gpu_start=2;
"

# screen -dmS c100-nt-trust-two bash -c "
    # source activate tag;
    # python trust2.py --n_rounds=250 --dba=1 --n_malicious=2 --col_size=2 --gpu_start=2;
# "
#
# screen -dmS c100-nt-trust-four bash -c "
    # source activate tag;
    # python trust2.py --n_rounds=250 --dba=1 --n_malicious=4 --col_size=2 --row_size=2 --gpu_start=2;
# "

