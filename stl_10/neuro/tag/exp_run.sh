screen -dmS stl-one bash -c "
    source activate tag;
    python train.py --n_rounds=250 --n_malicious=1 --alpha=1 --d_scale=1.5
"

screen -dmS stl-two bash -c "
    source activate tag;
    python train.py --n_rounds=250 --dba=1 --n_malicious=2 --alpha=1 --d_scale=1.5
"

screen -dmS stl-four bash -c "
    source activate tag;
    python train.py --n_rounds=250 --dba=1 --n_malicious=4 --alpha=1 --d_scale=1.5
"

