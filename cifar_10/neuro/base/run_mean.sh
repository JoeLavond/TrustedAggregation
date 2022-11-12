# screen -dmS vgg-meanone bash -c "
    # source activate joeenv;
    # python robust.py --n_rounds=250 --resnet=0 --trim_mean=1 --n_malicious=1 --gpu_start=2;
# "
#
# screen -dmS vgg-meantwo bash -c "
    # source activate joeenv;
    # python robust.py --n_rounds=250 --resnet=0 --trim_mean=1 --dba=1 --n_malicious=2 --col_size=2 --gpu_start=2;
# "
#
screen -dmS c10-mean-four bash -c "
    source activate joeenv;
    python robust.py --n_rounds=250 --resnet=0 --trim_mean=1 --dba=1 --n_malicious=4 --col_size=2 --row_size=2 --gpu_start=2;
"
