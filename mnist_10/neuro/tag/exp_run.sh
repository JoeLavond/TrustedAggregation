screen -dmS mnist-one bash -c "
    source activate tag;
    python train.py --n_rounds=250 --n_malicious=1 --gpu_start=0;
"

screen -dmS mnist-two bash -c "
    source activate tag;
    python train.py --n_rounds=250 --dba=1 --n_malicious=2 --gpu_start=1
"

screen -dmS mnist-four bash -c "
    source activate tag;
    python train.py --n_rounds=250 --dba=1 --n_malicious=4 --gpu_start=2
"
