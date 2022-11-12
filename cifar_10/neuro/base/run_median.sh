screen -dmS c10-median-one bash -c "
    source activate joeenv;
    python robust.py --n_rounds=250 --resnet=0 --trim_mean=0 --n_malicious=1 --gpu_start=2;
"

screen -dmS c10-median-two bash -c "
    source activate joeenv;
    python robust.py --n_rounds=250 --resnet=0 --trim_mean=0 --dba=1 --n_malicious=2 --col_size=2 --gpu_start=2;
"

screen -dmS c10-median-four bash -c "
    source activate joeenv;
    python robust.py --n_rounds=250 --resnet=0 --trim_mean=0 --dba=1 --n_malicious=4 --col_size=2 --row_size=2 --gpu_start=2;
"
