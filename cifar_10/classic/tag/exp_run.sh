screen -dmS vggtwo bash -c "
    conda deactivate
    source activate tag;
    python train.py --n_rounds=250 --resnet=0 --dba=1 --n_malicious=2 --col_size=2 --gpu_start=2 --d_scale=2.5;
"

screen -dmS vggfour bash -c "
    conda deactivate;
    source activate tag;
    python train.py --n_rounds=250 --resnet=0 --dba=1 --n_malicious=4 --col_size=2 --row_size=2 --gpu_start=2 --d_scale=2.5;
"

