""" Packages """
# base
import argparse
import numpy as np
import os
import re

# visual
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# source
import local_utils as lu


def get_args():

    parser = argparse.ArgumentParser()

    # path
    parser.add_argument('--data', default='cifar', type=str)
    parser.add_argument('--n_classes', default=100, type=int)

    parser.add_argument('--n_rounds', default=100, type=int)
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


def plot_threshold(
    data_val, data_user,  # ------- class distances for all users
    d_rounds,  # ------------------ output plot zoom
    m_start, n_malicious,  # ------ attack setting control
    path, suffix='',  # ----------- input and output location
    show=1  # --------------------- display plots?
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

    """ User Manipulation
    1. Seperate benign from malicious users
    2. Compute each round quantiles for summarizing values for plotting
    """
    # benign manipulation
    data_benign = data_user[data_user[:, 0] == 0, 1:]  # subset to benign users, remove malicious index column
    data_benign_r, data_benign_values = data_benign[:, 0], data_benign[:, 1:]  # seperate round column from data
    data_benign_max = data_benign_values.max(axis=1)
    data_benign_max_q1, data_benign_max_q3 = lu.get_quantiles(data_benign_r, data_benign_max)

    # clip to 0-1 scale
    benign_upper = data_benign_max_q3 + 1.5 * (data_benign_max_q3 - data_benign_max_q1)
    benign_upper = np.maximum(np.minimum(benign_upper, 1), 0)
    benign_lower = data_benign_max_q1 - 1.5 * (data_benign_max_q3 - data_benign_max_q1)
    benign_lower = np.maximum(np.minimum(benign_lower, 1), 0)

    # malicious users
    data_malicious = data_user[data_user[:, 0] == 1, 1:]  # subset to benign users, remove malicious index column
    data_malicious_r, data_malicious_values = data_malicious[:, 0], data_malicious[:, 1:]  # seperate round column from data
    data_malicious_max = data_malicious_values.max(axis=1)

    fig1, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True, figsize=(12, 4))

    # visual - benign
    ax1.set_title('Threshold vs. Benign')
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_xlabel('Communication Round')
    ax1.set_xlim(0, d_rounds)

    ax1.plot(data_val_r + 1, benign_upper, '--b')
    ax1.plot(data_val_r + 1, benign_lower, '--b')
    ax1.plot(data_val_r + 1, data_val_scaled_max_thresh, '-g')

    ax1.legend(labels=[
        'benign: q3 + 1.5 * IQR',
        'benign: q1 - 1.5 * IQR',
        'scaled threshold'
    ])

    # visual - malicious
    ax2.set_title('Threshold vs. Malicious')
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_xlabel('Communication Round')
    ax2.set_xlim(0, d_rounds)


    """ Malicious Users
    1. If multiple attacker, calculate quantiles
    2. Otherwise, plot single user points
    """
    if n_malicious > 1:


        # get lower and upper bound for malicious users
        data_malicious_max_q1, data_malicious_max_q3 = lu.get_quantiles(data_malicious_r, data_malicious_max)
        malicious_upper = data_malicious_max_q3 + 1.5 * (data_malicious_max_q3 - data_malicious_max_q1)
        malicious_upper = np.maximum(np.minimum(malicious_upper, 1), 0)
        malicious_lower = data_malicious_max_q1 - 1.5 * (data_malicious_max_q3 - data_malicious_max_q1)
        malicious_lower = np.maximum(np.minimum(malicious_lower, 1), 0)

        u_data_malicious_r = np.sort(np.unique(data_malicious_r))

        ax2.plot(u_data_malicious_r + 1, malicious_upper, '--r')
        ax2.plot(u_data_malicious_r + 1, malicious_lower, '--r')

        ax2.plot(data_val_r + 1, data_val_scaled_max_thresh, '-g')
        c_labels=[
            'malicious: q3 + 1.5 * IQR',
            'malicious: q1 - 1.5 * IQR',
            'scaled threshold'
        ]
        ax2.legend(labels=c_labels)

    else:

        u_data_malicious_r = np.sort(np.unique(data_malicious_r))
        ax2.plot(u_data_malicious_r + 1, data_malicious_max, '--r')
        ax2.plot(data_val_r + 1, data_val_scaled_max_thresh, '-g')

        c_labels=[
            'malicious',
            'scaled threshold'
        ]
        ax2.legend(labels=c_labels)


    """ Threshold diagnostics
    View running rates of acceptance for benign and malicious users
    """
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


""" Main function HERE """
def main():

    # setup
    args = get_args()
    if args.d_rounds is None:
        args.d_rounds = args.n_rounds

    path = os.path.join(
        f'/home/joe/03_federated/{args.data}_{args.n_classes}',
        ('neuro' if args.neuro else 'classic'),
        'tag',
        ('distributed' if args.dba else 'centralized'),
        f'alpha{args.alpha}--alpha_val{args.alpha_val}'
    )
    suffix = f'n_malicious{args.n_malicious}'

    if not os.path.exists(os.path.join(path, 'visuals')):
        os.makedirs(os.path.join(path, 'visuals'))


    """ Defense Plot - Thresholding """
    subdir = os.path.join(
        path,
        f'n_rounds{args.n_rounds}--d_start1--m_start1--n_malicious{args.n_malicious}'
    )

    temp_val = np.load(
        os.path.join(
            subdir,
            (
                'data/output_val_ks'
                + (f'--neuro_p{args.neuro_p}' if args.neuro else '')
                + '.npy'
            )
        ), allow_pickle=True
    )
    temp_user = np.load(
        os.path.join(
            subdir,
            (
                'data/output_user_ks'
                + (f'--neuro_p{args.neuro_p}' if args.neuro else '')
                + '.npy'
            )
        ), allow_pickle=True
    )

    plot_threshold(
        temp_val, temp_user,
        args.d_rounds,
        args.m_start, args.n_malicious,
        path, suffix,
        args.show
    )


if __name__ == "__main__":
    main()

