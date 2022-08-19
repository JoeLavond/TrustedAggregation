""" Packages """
# base
import argparse
import numpy as np
import os
import re

# visual
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
#matplotlib.pyplot.switch_backend('agg')
import seaborn as sns


def get_args():

    parser = argparse.ArgumentParser()

    # path
    parser.add_argument('--data', default='cifar', type=str)
    parser.add_argument('--n_classes', default=10, type=int)
    parser.add_argument('--n_rounds', default=100, type=int)
    parser.add_argument('--m_start', default=1, type=int)
    parser.add_argument('--n_malicious', default=1, type=int)
    parser.add_argument('--dba', default=0, type=int)

    # not intended to be modified
    parser.add_argument('--alpha', default=10000, type=int)
    parser.add_argument('--alpha_val', default=10000, type=int)
    parser.add_argument('--d_rounds', default=None, type=int)

    return parser.parse_args()


""" Helper functions """
def get_thresh_unscaled(x):

    out = []
    for i in range(len(x)):
        temp = x[:(i + 1)]
        out.append(np.mean(temp[np.argmin(temp):]))

    return out


def get_quantiles(r, values):

    u = np.unique(r)
    u.sort()

    q1, q3 = [], []
    for i in u:
        temp = values[r == i]
        temp_l = np.quantile(temp, .25)
        q1.append(temp_l)
        temp_u = np.quantile(temp, .75)
        q3.append(temp_u)

    return np.array(q1), np.array(q3)


def plot_threshold(
    data_val, data_user,
    warmup, d_rounds,
    path, suffix,
    n_malicious, m_start,
    n_classes
    ):

    # validation user
    data_val, data_user = np.array(data_val), np.array(data_user)
    data_val = data_val[:-1]  # remove last validation check for next iteration never run
    data_val, scaling = data_val[1:], data_val[0, 1:]  # remove scaling row, remove round from scaling
    data_val_r, data_val_values = data_val[:, 0], data_val[:, 1:]  # seperate round from data

    data_val_max = data_val_values.max(axis=1)
    data_val_max_thresh = get_thresh_unscaled(data_val_max)
    data_val_scaled = data_val_values * scaling
    data_val_scaled_max = data_val_scaled.max(axis=1)
    data_val_scaled_max_thresh = get_thresh_unscaled(data_val_scaled_max)
    data_val_scaled_max_thresh = np.maximum(np.minimum(data_val_scaled_max_thresh, 1), 0)

    # benign users
    data_benign = data_user[data_user[:, 0] == 0, 1:]  # subset to benign users, remove malicious index column
    data_benign_r, data_benign_values = data_benign[:, 0], data_benign[:, 1:]  # seperate round column from data
    data_benign_max = data_benign_values.max(axis=1)
    data_benign_max_q1, data_benign_max_q3 = get_quantiles(data_benign_r, data_benign_max)

    benign_upper = data_benign_max_q3 + 1.5 * (data_benign_max_q3 - data_benign_max_q1)
    benign_upper = np.maximum(np.minimum(benign_upper, 1), 0)
    benign_lower = data_benign_max_q1 - 1.5 * (data_benign_max_q3 - data_benign_max_q1)
    benign_lower = np.maximum(np.minimum(benign_lower, 1), 0)

    # malicious users
    data_malicious = data_user[data_user[:, 0] == 1, 1:]  # subset to benign users, remove malicious index column
    data_malicious_r, data_malicious_values = data_malicious[:, 0], data_malicious[:, 1:]  # seperate round column from data
    data_malicious_max = data_malicious_values.max(axis=1)

    fig1, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(8, 4))

    # visual - benign
    ax1.set_title('Threshold vs. Benign')
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_xlabel('Communication Round')
    ax1.set_xlim(0, d_rounds)

    ax1.plot(data_val_r + 1, benign_upper, '--b')
    ax1.plot(data_val_r + 1, benign_lower, '--b')

    # warmup
    if warmup > 0:

        round_scalings = 2 * np.ones_like(data_val_scaled_max_thresh)
        temp = np.linspace(
            start=(n_classes + 1) / n_classes,
            stop=2,
            num=warmup
        )

        round_scalings[:len(temp)] = temp
        ax1.plot(data_val_r + 1, round_scalings * data_val_scaled_max_thresh, '-g')

    else:

        round_scalings = 2
        ax1.plot(data_val_r + 1, round_scalings * data_val_scaled_max_thresh, '-g')

    ax1.legend(labels=[
        'benign max ks: q3 + 1.5 * IQR',
        'benign max ks: q1 - 1.5 * IQR',
        'scaled threshold'
    ])
    ax1.vlines(m_start, 0, 1, colors='r')
    ax1.text(m_start, 1.033, 'attack start', c='r')

    # visual - malicious
    ax2.set_title('Threshold vs. Malicious')
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_xlabel('Communication Round')
    ax2.set_xlim(0, d_rounds)

    if n_malicious > 1:


        # get lower and upper bound for malicious users
        data_malicious_max_q1, data_malicious_max_q3 = get_quantiles(data_malicious_r, data_malicious_max)
        malicious_upper = data_malicious_max_q3 + 1.5 * (data_malicious_max_q3 - data_malicious_max_q1)
        malicious_upper = np.maximum(np.minimum(malicious_upper, 1), 0)
        malicious_lower = data_malicious_max_q1 - 1.5 * (data_malicious_max_q3 - data_malicious_max_q1)
        malicious_lower = np.maximum(np.minimum(malicious_lower, 1), 0)

        u_data_malicious_r = np.sort(np.unique(data_malicious_r))

        ax2.plot(u_data_malicious_r + 1, malicious_upper, '--r')
        ax2.plot(u_data_malicious_r + 1, malicious_lower, '--r')


        # warmup
        if warmup > 0:

            round_scalings = 2 * np.ones_like(data_val_scaled_max_thresh)
            temp = np.linspace(
                start=(n_classes + 1) / n_classes,
                stop=2,
                num=warmup
            )

            round_scalings[:len(temp)] = temp
            ax2.plot(data_val_r + 1, round_scalings * data_val_scaled_max_thresh, '-g')

        else:

            round_scalings = 2
            ax2.plot(data_val_r + 1, round_scalings * data_val_scaled_max_thresh, '-g')


        c_labels=[
            'malicious max ks: q3 + 1.5 * IQR',
            'malicious max ks: q1 - 1.5 * IQR',
            'scaled threshold'
        ]
        ax2.legend(labels=c_labels)

    else:

        u_data_malicious_r = np.sort(np.unique(data_malicious_r))
        ax2.plot(u_data_malicious_r + 1, data_malicious_max, '--r')

        # warmup
        if warmup > 0:

            round_scalings = 2 * np.ones_like(data_val_scaled_max_thresh)
            temp = np.linspace(
                start=(n_classes + 1) / n_classes,
                stop=2,
                num=warmup
            )

            round_scalings[:len(temp)] = temp
            ax2.plot(data_val_r + 1, round_scalings * data_val_scaled_max_thresh, '-g')

        else:

            round_scalings = 2
            ax2.plot(data_val_r + 1, round_scalings * data_val_scaled_max_thresh, '-g')

        c_labels=[
            'malicious max ks',
            'scaled threshold'
        ]
        ax2.legend(labels=c_labels)

    ax2.vlines(m_start, 0, 1, colors='r')
    ax2.text(m_start, 1.033, 'attack start', c='r')

    if path is not None:
        plt.savefig(os.path.join(path, 'visuals', f'threshold{suffix}--d_rounds{d_rounds}.png'), bbox_inches='tight')

    #plt.show()
    plt.close()


""" Main function HERE """
def main():

    # setup
    args = get_args()
    if args.d_rounds is None:
        args.d_rounds = args.n_rounds

    path = os.path.join(
        f'/home/joe/03_federated/{args.data}-{args.n_classes}/our',
        ('distributed' if args.dba else 'centralized'),
        f'alpha{args.alpha}--alpha_val{args.alpha_val}'
    )
    suffix = f'--dba{args.dba}'

    if not os.path.exists(os.path.join(path, 'visuals')):
        os.makedirs(os.path.join(path, 'visuals'))


    """ Defense Plot - Thresholding """
    subdir = os.path.join(
        path,
        f'n_rounds{args.n_rounds}--d_start1--m_start1--n_malicious{args.n_malicious}'
    )

    temp_val = np.load(os.path.join(subdir, 'data/output_val_ks.npy'), allow_pickle=True)
    temp_user = np.load(os.path.join(subdir, 'data/output_user_ks.npy'), allow_pickle=True)

    plot_threshold(
        temp_val, temp_user,
        args.warmup, args.d_rounds,
        path, suffix,
        args.n_malicious, args.m_start,
        args.n_classes
    )


if __name__ == "__main__":
    main()
