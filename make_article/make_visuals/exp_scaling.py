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

    parser.add_argument('--data', default='cifar', type=str)
    parser.add_argument('--n_classes', default=100, type=int)
    parser.add_argument('--d_rounds', default=None, type=int)
    parser.add_argument('--show', default=1, type=int)

    parser.add_argument('--n_rounds', default=100, type=int)
    parser.add_argument('--n_malicious', default=1, type=int)
    parser.add_argument('--m_start', default=1, type=int)
    parser.add_argument('--dba', default=0, type=int)

    parser.add_argument('--neuro', default=0, type=int)
    parser.add_argument('--neuro_p', default=0.1, type=float)

    parser.add_argument('--alpha', default=10000, type=int)
    parser.add_argument('--alpha_val', default=10000, type=int)

    return parser.parse_args()


""" Scaling figure """
def plot_scaling(
    data_val, data_user,  # ------ class distances for all users
    d_rounds,  # ----------------- output plot zoom
    path, suffix='',  # ---------- input and output location
    show=1  # -------------------- display plots?
    ):

    """ Documentation
    Function: Create plot for understanding effectiveness of class frequency scaling
    1. Validation threshold w/ and w/o class frequency scaling
    2. Benign user upper and lower bounds on max class distance
    Usage: Visualize importance of scaling for article
    """

    """ Validation Manipulation
    1. Last row for next round threshold - never used
    2. First row includes scaling information - round index -1
    3. Compute threshold w/ and w/o class frequency scaling
    """
    data_val, data_user = np.array(data_val), np.array(data_user)
    data_val = data_val[:-1]  # remove last validation check for next iteration never run
    data_val, scaling = data_val[1:], data_val[0, 1:]  # remove scaling row, remove round from scaling
    data_val_r, data_val_values = data_val[:, 0], data_val[:, 1:]  # seperate round from data

    # threshold without class frequency scaling
    data_val_max = data_val_values.max(axis=1)
    data_val_max_thresh = lu.min_mean_smooth(data_val_max, scale=2)

    # with class frequency scaling
    data_val_scaled = data_val_values * scaling
    data_val_scaled_max = data_val_scaled.max(axis=1)
    data_val_scaled_max_thresh = lu.min_mean_smooth(data_val_scaled_max, scale=2)

    data_val_max_thresh = np.maximum(np.minimum(data_val_max_thresh, 1), 0)
    data_val_scaled_max_thresh = np.maximum(np.minimum(data_val_scaled_max_thresh, 1), 0)

    """ Benign Manipulation
    1. Subset to benign from all users
    2. Compute each round quantiles for summarizing values for plotting
    """
    data_benign = data_user[data_user[:, 0] == 0, 1:]  # subset to benign users, remove malicious index column
    data_benign_r, data_benign_values = data_benign[:, 0], data_benign[:, 1:]  # seperate round column from data
    data_benign_max = data_benign_values.max(axis=1)
    data_benign_max_q1, data_benign_max_q3 = lu.get_quantiles(data_benign_r, data_benign_max)

    # quantile computation
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

    plt.plot(data_val_r + 1, data_val_max_thresh, '-r')
    plt.plot(data_val_r + 1, data_val_scaled_max_thresh, '-g')

    plt.legend(labels=[
        'benign: q3 + 1.5 * IQR',
        'benign: q1 - 1.5 * IQR',
        'unscaled threshold',
        'scaled threshold'
    ])

    if path is not None:
        plt.savefig(
            os.path.join(
                path, 'visuals', f'scaling{suffix}--d_rounds{d_rounds}.png'
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
        'alpha' + str(args.alpha) + '--alpha_val' + str(args.alpha_val)
    )

    if not os.path.exists(os.path.join(path, 'visuals')):
        os.makedirs(os.path.join(path, 'visuals'))


    """ Defense Plot - Scaling """
    # import no attack nor defense data
    to_reads = [
        f.path for f in os.scandir(path) if re.search(
            f'n_rounds{args.n_rounds}--d_start{args.n_rounds + 1}--m_start{args.n_rounds + 1}', f.path
        )
    ]

    temp_val = np.load(
        os.path.join(
            to_reads[0],
            (
                'data/output_global_acc'
                + (f'--neuro_p{args.neuro_p}' if args.neuro else '')
                + '.npy'
            )
        ), allow_pickle=True
    )
    temp_user = np.load(
        os.path.join(
            to_reads[0],
            (
                'data/output_user_ks'
                + (f'--neuro_p{args.neuro_p}' if args.neuro else '')
                + '.npy'
            )
        ), allow_pickle=True
    )

    plot_scaling(
        temp_val,
        temp_user,
        args.d_rounds,
        path,
        show=args.show
    )


if __name__ == "__main__":
    main()

