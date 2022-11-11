screen -dmS atktwo bash -c "
    conda deactivate
    source activate joeenv;
    python trust.py --n_rounds=250 --dba=1 --n_malicious=2 --col_size=2 --gpu_start=2;
"

screen -dmS atkfour bash -c "
    conda deactivate;
    source activate joeenv;
    python trust.py --n_rounds=250 --dba=1 --n_malicious=4 --col_size=2 --row_size=2 --gpu_start=2;
"

