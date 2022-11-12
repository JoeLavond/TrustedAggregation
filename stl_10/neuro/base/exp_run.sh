screen -dmS s10-nt-trustone bash -c "
    source activate joeenv;
    python trust.py --n_rounds=250 --n_malicious=1 --gpu_start=2;
"
