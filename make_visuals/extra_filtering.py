""" Packages """
# base
import argparse
import numpy as np
import os
from pathlib import Path
import re

# visual
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# source
import local_utils as lu


def get_args():
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument('--data', default='cifar', type=str)
    parser.add_argument('--n_classes', default=10, type=int)

    parser.add_argument('--n_rounds', default=250, type=int)
    parser.add_argument('--m_start', default=1, type=int)

    parser.add_argument('--n_malicious', default=1, type=int)
    parser.add_argument('--dba', default=0, type=int)

    parser.add_argument('--neuro', default=0, type=int)
    parser.add_argument('--neuro_p', default=0.1, type=float)

    # not intended to be modified
    parser.add_argument('--alpha', default=10000, type=int)
    parser.add_argument('--alpha_val', default=10000, type=int)

    parser.add_argument('--d_rounds', default=None, type=int)
    parser.add_argument('--show', default=1, type=int)

    return parser.parse_args()


def preprocessing_for_model_filtering(
        data_val, data_user,  # ------- class distances for all users
):
    """ Documentation
    Function: Create plot for understanding threshold performance
    1. Validation threshold w/ scaling and smoothing
    2. Benign user upper and lower bounds on max class distance
    2. Malicious user upper and lower bounds on max class distance
        - Quantile if multiple attackers
        - Line plot if single attacker
    Usage: Visualize importance of scaling for article
    """

    """ Validation Manipulation
    1. Last row for next round threshold - never used
    2. First row includes scaling information - round index -1
    3. Compute threshold
    """
    data_val, data_user = np.array(data_val), np.array(data_user)
    data_val = data_val[:-1]  # remove last validation check for next iteration never run
    data_val, scaling = data_val[1:], data_val[0, 1:]  # remove scaling row, remove round from scaling
    data_val_r, data_val_values = data_val[:, 0], data_val[:, 1:]  # seperate round from data

    data_val_scaled = data_val_values * scaling
    data_val_scaled_max = data_val_scaled.max(axis=1)
    data_val_scaled_max_thresh = lu.min_mean_smooth(data_val_scaled_max, scale=2)
    data_val_scaled_max_thresh = np.minimum(data_val_scaled_max_thresh, 1)

    data_val = pd.DataFrame({
        'round': data_val_r,
        'threshold': data_val_scaled_max_thresh
    })

    """ User Manipulation
    1. Seperate benign from malicious users
    2. Compute each round quantiles for summarizing values for plotting
    """
    # benign manipulation
    data_user_malicious, data_user_r, data_user_values = (
        data_user[:, 0],
        data_user[:, 1],
        data_user[:, 2:]  # seperate round column from data
    )
    data_user_max = data_user_values.max(axis=1)
    data_user_accepted = np.array([
        v < data_val_scaled_max_thresh[int(r - 1)]
        for r, v in zip(data_user_r, data_user_max)
    ])

    data_user = pd.DataFrame({
        'malicious': data_user_malicious,
        'round': data_user_r,
        'max': data_user_max,
        'accepted': data_user_accepted
    })

    return data_val, data_user


""" Main function HERE """


def main():
    # setup
    args = get_args()
    if args.d_rounds is None:
        args.d_rounds = args.n_rounds

    path = os.path.join(
        f'{Path.home()}',
        'Documents',
        'TAG',
        f'{args.data}_{args.n_classes}',
        ('neuro' if args.neuro else 'classic'),
        'tag',
        ('distributed' if args.dba else 'centralized'),
        f'alpha{args.alpha}--alpha_val{args.alpha_val}'
    )
    suffix = f'--n_malicious{args.n_malicious}'

    if not os.path.exists(os.path.join(path, 'visuals')):
        os.makedirs(os.path.join(path, 'visuals'))

    """ Defense Plot - Filtering """
    subdir = os.path.join(
        path,
        f'n_rounds{args.n_rounds}--d_start1--m_start1--n_malicious{args.n_malicious}'
    )

    temp_val_ks = np.load(
        os.path.join(
            subdir,
            'data',
            (
                    'output_val_ks'
                    + (f'--neuro_p{args.neuro_p}' if args.neuro else '')
                    + '.npy'
            ),
        ), allow_pickle=True
    )
    temp_user_ks = np.load(
        os.path.join(
            subdir,
            'data',
            (
                    'output_user_ks'
                    + (f'--neuro_p{args.neuro_p}' if args.neuro else '')
                    + '.npy'
            )
        ), allow_pickle=True
    )

    data_val, data_user = preprocessing_for_model_filtering(
        temp_val_ks, temp_user_ks
    )

    temp_user = np.load(
        os.path.join(
            subdir,
            'data',
            (
                    'output_user'
                    + (f'--neuro_p{args.neuro_p}' if args.neuro else '')
                    + '.npy'
            )
        ), allow_pickle=True
    )

    # ensure that user data is the same
    assert np.equal(data_user['malicious'].values, temp_user[:, 0]).all(), 'malicious'
    assert np.equal(data_user['round'].values, temp_user[:, 1]).all(), 'round'

    # combine user id
    data_user['id'] = temp_user[:, 2]

    """
    fig1, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True, figsize=(12, 4))

    # diagnostic computation
    (_, benign_diag_run) = lu.threshold_diagnostics(
        data_benign_r,
        data_benign_max,
        data_val_scaled_max_thresh
    )

    (temp, malicious_diag_run) = lu.threshold_diagnostics(
        data_malicious_r,
        data_malicious_max,
        data_val_scaled_max_thresh
    )

    # visual
    ax3.set_title('Model Filtering')
    ax3.plot(
        range(1, len(data_val_scaled_max_thresh) + 1),
        benign_diag_run, color='b'
    )
    ax3.plot(
        range(m_start, len(data_val_scaled_max_thresh) + 1),
        malicious_diag_run, color='r'
    )
    ax3.legend(labels=['benign users', 'malicious users'])
    ax3.set_xlim(0, d_rounds)
    ax3.set_xlabel('Communication Round')
    ax3.set_ylim(-0.05, 1.1)
    ax3.set_ylabel('Acceptance Rate')


    if path is not None:
        plt.savefig(
            os.path.join(
                path, 'visuals', f'threshold{suffix}--d_rounds{d_rounds}.png'
            ),
            bbox_inches='tight'
        )

    if show:
        plt.show()
    else:
        plt.close()
    """


if __name__ == "__main__":
    main()
