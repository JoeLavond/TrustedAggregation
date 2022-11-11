screen -dmS c100ntatkone bash -c "
    conda deactivate
    source activate joeenv;
    python trust.py --n_rounds=250 --n_malicious=1 --gpu_start=0;
"

screen -dmS c100ntatktwo bash -c "
    conda deactivate
    source activate joeenv;
    python trust.py --n_rounds=250 --dba=1 --n_malicious=2 --col_size=2 --gpu_start=0;
"

screen -dmS c100ntatkfour bash -c "
    conda deactivate;
    source activate joeenv;
    python trust.py --n_rounds=250 --dba=1 --n_malicious=4 --col_size=2 --row_size=2 --gpu_start=0;
"

