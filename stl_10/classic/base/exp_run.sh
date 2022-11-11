screen -dmS s10atkone bash -c "
    conda deactivate
    source activate joeenv;
    python trust.py --n_rounds=250 --n_malicious=1 --gpu_start=2;
"

screen -dmS s10atkfour bash -c "
    conda deactivate;
    source activate joeenv;
    python trust.py --n_rounds=250 --dba=1 --n_malicious=4 --col_size=12 --row_size=12 --gpu_start=2;
"

