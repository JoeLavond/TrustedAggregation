# screen -dmS nt-mu-001 bash -c "
    # source activate joeenv;
    # python train.py --n_rounds=250 --mu=.001 --gpu_start=0;
# "
#
# screen -dmS nt-mu-005 bash -c "
    # source activate joeenv;
    # python train.py --n_rounds=250 --mu=.005 --gpu_start=0;
# "

screen -dmS nt-mu-01 bash -c "
    source activate joeenv;
    python train.py --n_rounds=250 --mu=.01 --gpu_start=2;
"

# screen -dmS nt-two-mu-001 bash -c "
    # source activate joeenv;
    # python train.py --n_rounds=250 --mu=.001 --dba=1 --n_malicious=2 --col_size=2 --gpu_start=2;
# "

screen -dmS nt-two-mu-005 bash -c "
    source activate joeenv;
    python train.py --n_rounds=250 --mu=.005 --dba=1 --n_malicious=2 --col_size=2 --gpu_start=0;
"

screen -dmS nt-two-mu-01 bash -c "
    source activate joeenv;
    python train.py --n_rounds=250 --mu=.01 --dba=1 --n_malicious=2 --col_size=2 --gpu_start=0;
"

