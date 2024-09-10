screen -dmS cifar-one-one-five bash -c "
    source activate tag;
    python train.py --n_rounds=250 --n_malicious=1 --alpha=1 --d_scale=1.5
"

screen -dmS cifar-two-one-five bash -c "
    source activate tag;
    python train.py --n_rounds=250 --dba=1 --n_malicious=2 --alpha=1 --d_scale=1.5
"

screen -dmS cifar-four-one-five bash -c "
    source activate tag;
    python train.py --n_rounds=250 --dba=1 --n_malicious=4 --alpha=1 --d_scale=1.5
"

