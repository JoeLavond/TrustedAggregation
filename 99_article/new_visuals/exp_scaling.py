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
    parser.add_argument('--root', default='02_warmup', type=str)
    parser.add_argument('--n_classes', default=10, type=int)
    parser.add_argument('--dba', default=0, type=int)
    parser.add_argument('--n_malicious', default=1, type=int)
    parser.add_argument('--n_rounds', default=100, type=int)
    parser.add_argument('--m_start', default=1, type=int)
    parser.add_argument('--alpha', default=10000, type=int)
    parser.add_argument('--alpha_val', default=10000, type=int)
    parser.add_argument('--warmup', default=0, type=int)
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


def plot_scaling(
    data_val, data_user,
    warmup, d_rounds,
    path, suffix,
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

    data_val_max_thresh = np.maximum(np.minimum(data_val_max_thresh, 1), 0)
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

    # visual
    plt.figure()
    plt.xlabel('Communication Round')
    plt.xlim(0, d_rounds)

    plt.plot(data_val_r + 1, benign_upper, '--b')
    plt.plot(data_val_r + 1, benign_lower, '--b')

    plt.plot(data_val_r + 1, 2 * data_val_max_thresh, '-r')
    plt.plot(data_val_r + 1, 2 * data_val_scaled_max_thresh, '-g')

    plt.legend(labels=[
        'benign max ks: q3 + 1.5 * IQR',
        'benign max ks: q1 - 1.5 * IQR',
        'unscaled threshold',
        'scaled threshold'
    ])

    if path is not None:
        plt.savefig(os.path.join(path, 'visuals', f'scaling{suffix}--d_rounds{d_rounds}.png'), bbox_inches='tight')

    plt.show()
    plt.close()


""" Main function HERE """
def main():

    # setup
    args = get_args()
    if args.d_rounds is None:
        args.d_rounds = args.n_rounds

    path = os.path.join(
        f'/home/joe/03_federated/{args.root}',
        ('distributed' if args.dba else 'centralized'),
        'alpha' + str(args.alpha) + '--alpha_val' + str(args.alpha_val)
    )
    suffix = '--n_malicious' + str(args.n_malicious) + '--m_start' + str(args.m_start)

    if not os.path.exists(os.path.join(path, 'visuals')):
        os.makedirs(os.path.join(path, 'visuals'))

    # experiments
    titles = ['No Attack or Defense', 'Attack Only', 'Defense Only', 'Attack and Defense']
    values = [
        (args.n_rounds + 1, args.n_rounds + 1),  # no attack or defense
        (args.n_rounds + 1, args.m_start),  # attack only
        (1, args.n_rounds + 1),  # defense only
        (1, args.m_start)  # attack and defense
   ]


    """ Defense Plot - Scaling """
    # import no attack nor defense data
    (i, j) = values[0]
    to_reads = [
        f.path for f in os.scandir(path) if re.search(
            f'n_rounds{args.n_rounds}--d_start{args.n_rounds + 1}--m_start{args.n_rounds + 1}', f.path
        )
    ]

    temp_val = np.load(os.path.join(to_reads[0], 'data/output_val_ks.npy'), allow_pickle=True)
    temp_user = np.load(os.path.join(to_reads[0], 'data/output_user_ks.npy'), allow_pickle=True)

    plot_scaling(
        temp_val, temp_user,
        args.warmup, args.d_rounds,
        path, suffix,
        args.n_classes
    )


if __name__ == "__main__":
    main()
