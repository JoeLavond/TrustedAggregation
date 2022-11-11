screen -dmS vggtwo bash -c "
    conda deactivate
    source activate joeenv;
    python train.py --n_rounds=250 --alpha=1 --alpha_val=1 --d_scale=1.1 --dba=1 --n_malicious=2 --col_size=2 --gpu_start=2;
"

screen -dmS vggfour bash -c "
    conda deactivate;
    source activate joeenv;
    python train.py --n_rounds=250 ---alpha=1 --alpha_val=1 --d_scale=1.1 -dba=1 --n_malicious=4 --col_size=2 --row_size=2 --gpu_start=2;
"

