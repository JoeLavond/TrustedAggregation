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

    # find root dir
    parser.add_argument('--data', default='cifar', type=str)
    parser.add_argument('--n_classes', default=10, type=int)
    parser.add_argument('--n_rounds', default=100, type=int)
    # control output
    parser.add_argument('--d_rounds', default=30, type=int)

    return parser.parse_args()


def plot_threshold(
    data_val,  # ------ input threshold object
    d_rounds,  # ------ input data and rounds to display
    path='.',  # ------ specify plot location
    suffix='',  # ----- modify output name
    window=3  # ------- window to compute moving average over
    ):

    """
    Function: Return comparison of smoothing techniques to global min mean
    Usage: Figure to explain our proposed method for article
    """

    # validation user
    data_val = np.array(data_val)
    data_val = data_val[:-1]  # remove last validation for iteration never run
    data_val, scaling = data_val[1:], data_val[0, 1:]  # remove scaling row, remove round from scaling
    data_val_r, data_val_values = data_val[:, 0], data_val[:, 1:]  # seperate round from data

    data_val_max = data_val_values.max(axis=1)
    data_val_max_thresh = lu.min_mean_smooth(data_val_max, scale=2)
    data_val_max_thresh = np.minimum(data_val_max_thresh, 1)  # clip threshold to valid range
    (moving_avg, exp_smooth) = lu.get_smoothing(2 * data_val_max, window)

    # create visual
    plt.figure()
    plt.xlabel('Communication Round')
    plt.xlim(0, d_rounds)  # zoom to rounds of interest
    plt.ylim(-0.05, 1.1)

    plt.plot(data_val_r + 1, np.minimum(2 * data_val_max, 1), '-k')
    plt.plot(data_val_r + 1, np.minimum(moving_avg, 1), '--r')
    plt.plot(data_val_r + 1, np.minimum(exp_smooth, 1), '--', color='orange')
    plt.plot(data_val_r + 1, np.minimum(data_val_max_thresh, 1), '--b')

    plt.legend(labels=[
        'base threshold',
        f'moving average (window={window})',
        'exponential smoothing',
        'global min-mean smoothing'
    ])

    if path is not None:
        plt.savefig(
            os.path.join(
                path,
                f'smoothing{suffix}--d_rounds{d_rounds}.png'
            ),
            bbox_inches='tight'
        )

    plt.show()
    plt.close()


""" Main function HERE """
def main():

    # setup
    args = get_args()
    if args.d_rounds is None:
        args.d_rounds = args.n_rounds

    path = os.path.join(
        '/home/joe/03_federated',
        f'{args.data}-{args.n_classes}/our/centralized',
        'alpha10000--alpha_val10000'
    )

    if not os.path.exists('visuals'):
        os.mkdir('visuals')


    """ Defense Plot - Thresholding """
    # import no attack nor defense data
    subdir = os.path.join(
        path,
        f'n_rounds{args.n_rounds}--d_start1--m_start1--n_malicious1'
    )

    temp_val = np.load(os.path.join(subdir, 'data/output_val_ks.npy'), allow_pickle=True)
    plot_threshold(temp_val, args.d_rounds)


if __name__ == "__main__":
    main()

