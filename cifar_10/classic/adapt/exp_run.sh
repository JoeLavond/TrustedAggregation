# screen -dmS mu001-scale11 bash -c "
    # source activate joeenv;
    # python train.py --n_rounds=250 --mu=.001 --gpu_start=0 --d_scale=1.1;
# "
#
# screen -dmS mu005-scale11 bash -c "
    # source activate joeenv;
    # python train.py --n_rounds=250 --mu=.005 --gpu_start=0 --d_scale=1.1;
# "
#
# screen -dmS mu01-scale11 bash -c "
    # source activate joeenv;
    # python train.py --n_rounds=250 --mu=.01 --gpu_start=0 --d_scale=1.1;
# "
#
# screen -dmS two-mu001-scale11 bash -c "
    # source activate joeenv;
    # python train.py --n_rounds=250 --mu=.001 --dba=1 --n_malicious=2 --col_size=2 --d_scale=1.1 --gpu_start=0;
# "
#
# screen -dmS two-mu005-scale11 bash -c "
    # source activate joeenv;
    # python train.py --n_rounds=250 --mu=.005 --dba=1 --n_malicious=2 --col_size=2 --d_scale=1.1 --gpu_start=0;
# "
#
# screen -dmS two-mu01-scale11 bash -c "
    # source activate joeenv;
    # python train.py --n_rounds=250 --mu=.01 --dba=1 --n_malicious=2 --col_size=2 --d_scale=1.1 --gpu_start=0;
# "
#
# screen -dmS four-mu001-scale11 bash -c "
    # source activate joeenv;
    # python train.py --n_rounds=250 --mu=.001 --dba=1 --n_malicious=4 --col_size=2 --row_size=2 --d_scale=1.1 --gpu_start=0;
# "
#
# screen -dmS four-mu005-scale11 bash -c "
    # source activate joeenv;
    # python train.py --n_rounds=250 --mu=.005 --dba=1 --n_malicious=4 --col_size=2 --row_size=2 --d_scale=1.1 --gpu_start=0;
# "
#
# screen -dmS four-mu01-scale11 bash -c "
    # source activate joeenv;
    # python train.py --n_rounds=250 --mu=.01 --dba=1 --n_malicious=4 --col_size=2 --row_size=2 --d_scale=1.1 --gpu_start=0;
# "
#
