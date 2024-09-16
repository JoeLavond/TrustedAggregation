# screen -dmS nt-mu001-scale11 bash -c "
    # source activate tag;
    # python train.py --n_rounds=250 --mu=.001 --d_scale=1.1 --gpu_start=2;
# "
#
# screen -dmS nt-mu005-scale11 bash -c "
    # source activate tag;
    # python train.py --n_rounds=250 --mu=.005 --d_scale=1.1 --gpu_start=0;
# "
#
# screen -dmS nt-mu01-scale11 bash -c "
    # source activate tag;
    # python train.py --n_rounds=250 --mu=.01 --d_scale=1.1 --gpu_start=0;
# "
#
# screen -dmS nt-two-mu001-scale11 bash -c "
    # source activate tag;
    # python train.py --n_rounds=250 --mu=.001 --dba=1 --n_malicious=2 --col_size=2 --d_scale=1.1 --gpu_start=0;
# "
#
# screen -dmS nt-two-mu005-scale11 bash -c "
    # source activate tag;
    # CUDA_VISIBLE_DEVICES=0,2 python train.py --n_rounds=250 --mu=.005 --dba=1 --n_malicious=2 --col_size=2 --d_scale=1.1 --gpu_start=0;
# "
#
# screen -dmS nt-two-mu01-scale11 bash -c "
    # source activate tag;
    # python train.py --n_rounds=250 --mu=.01 --dba=1 --n_malicious=2 --col_size=2 --d_scale=1.1 --gpu_start=0;
# "
#
# screen -dmS nt-four-mu001-scale11 bash -c "
    # source activate tag;
    # python train.py --n_rounds=250 --mu=.001 --dba=1 --n_malicious=4 --row_size=2 --col_size=2 --d_scale=1.1 --gpu_start=2;
# "
#
# screen -dmS nt-four-mu005-scale11 bash -c "
    # source activate tag;
    # python train.py --n_rounds=250 --mu=.005 --dba=1 --n_malicious=4 --row_size=2 --col_size=2 --d_scale=1.1 --gpu_start=2;
# "
#
# screen -dmS nt-four-mu01-scale11 bash -c "
    # source activate tag;
    # python train.py --n_rounds=250 --mu=.01 --dba=1 --n_malicious=4 --row_size=2 --col_size=2 --d_scale=1.1 --gpu_start=2;
# "
#
