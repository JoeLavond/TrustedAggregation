screen -dmS cifar-one bash -c "
    source activate tag;
    python train.py --n_rounds=1 --n_malicious=1 --alpha=1
"

screen -dmS cifar-two bash -c "
    source activate tag;
    python train.py --n_rounds=1 --dba=1 --n_malicious=2 --alpha=1
"

screen -dmS cifar-four bash -c "
    source activate tag;
    python train.py --n_rounds=1 --dba=1 --n_malicious=4 --alpha=1
"

