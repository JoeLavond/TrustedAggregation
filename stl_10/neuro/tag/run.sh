screen -dmS stl10-tag-one bash -c """
    source activate tag;
    python train.py --n_rounds=250 --n_malicious=1 --col_size=12 --row_size=12 --d_scale=2;
"""

