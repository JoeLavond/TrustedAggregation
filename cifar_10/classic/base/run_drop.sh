screen -dmS drop- bash -c "
    source activate joeenv;
    python drop.py --n_rounds=250 --dba=0 --n_malicious=2 --drop_p=0.25 --gpu_start=0;
"

screen -dmS drop- bash -c "
    source activate joeenv;
    python robust.py --n_rounds=250 --resnet=0 --trim_mean=1 --dba=1 --n_malicious=2 --col_size=2 --gpu_start=0;
"

screen -dmS drop- bash -c "
    source activate joeenv;
    python robust.py --n_rounds=250 --resnet=0 --trim_mean=1 --dba=1 --n_malicious=4 --col_size=2 --row_size=2 --gpu_start=0;
"
