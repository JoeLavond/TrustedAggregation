screen -dmS vgg-nt-one bash -c "
    source activate tag;
    python train.py --n_rounds=250 --resnet=0 --n_malicious=1 --gpu_start=0 --d_scale=2.5;
"

screen -dmS vgg-nt-two bash -c "
    source activate tag;
    python train.py --n_rounds=250 --resnet=0 --dba=1 --n_malicious=2 --col_size=2 --gpu_start=2 --d_scale=2.5;
"

screen -dmS vgg-nt-four bash -c "
    source activate tag;
    python train.py --n_rounds=250 --resnet=0 --dba=1 --n_malicious=4 --col_size=2 --row_size=2 --gpu_start=2 --d_scale=2.5;
"

