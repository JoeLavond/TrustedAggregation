screen -dmS c100-nt-trustone bash -c "
    source activate joeenv;
    python trust.py --n_rounds=250 --resnet=0 --n_malicious=1 --gpu_start=2;
"

screen -dmS c100-nt-trusttwo bash -c "
    source activate joeenv;
    python trust.py --n_rounds=250 --resnet=0 --dba=1 --n_malicious=2 --col_size=2 --gpu_start=2;
"

screen -dmS c100-nt-trustfour bash -c "
    urce activate joeenv;
    python trust.py --n_rounds=250 --resnet=0 --dba=1 --n_malicious=4 --col_size=2 --row_size=2 --gpu_start=2;
"

