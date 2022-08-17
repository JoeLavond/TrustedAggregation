""" Packages """
# base
import argparse
import numpy as np
import os
import re

# smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

# visual
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
#matplotlib.pyplot.switch_backend('agg')
import seaborn as sns


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_rounds', default=250, type=int)
    parser.add_argument('--d_rounds', default=30, type=int)
    parser.add_argument('--alpha', default=10000, type=int)
    parser.add_argument('--alpha_val', default=10000, type=int)

    return parser.parse_args()


""" Helper functions """
def get_thresh(x):

    out = []
    for i in range(len(x)):
        temp = x[:(i + 1)]
        out.append(2 * np.mean(temp[np.argmin(temp):]))

    return out

def get_quantiles(r, values, running=0, global_min=0):

    u = np.unique(r)
    u.sort()

    q1, q3 = [], []
    for i in u:

        if running:
            index = (r <= i)
            temp = values[index]

            if global_min:

                # find min of values in this or previous round
                min_loc = np.argmin(temp)
                min_r = r[index][min_loc]

                index = np.logical_and(
                    index, r >= min_r  # occur since global min
                )
                temp = values[index]

        else:
            index = (r == i)
            temp = values[index]

        # calculate quantiles
        temp_l = np.quantile(temp, .25)
        q1.append(temp_l)
        temp_u = np.quantile(temp, .75)
        q3.append(temp_u)

    return np.array(q1), np.array(q3)


def get_smoothing(x, w=3):

    out_simple, out_exp = [], []
    for i in range(len(x)):
        temp = x[:(i + 1)]

        if i == 0:
            out_simple.append(x[0])
            out_exp.append(x[0])

        else:
            out_simple.append(np.mean(x[max([0, i - (w - 1)]):(i + 1)]))

            exp_smooth = ExponentialSmoothing(temp, trend='mul', initialization_method="estimated").fit()
            out_exp.append(exp_smooth.fittedvalues[-1])

    return (np.array(out_simple), np.array(out_exp))


def plot_threshold(
    data_val, d_rounds,  # ----- input data and rounds to display
    suffix, path='.'  # -------- .png output location
    ):

    # validation user
    data_val = np.array(data_val)
    data_val = data_val[:-1]  # remove last validation check for next iteration never run
    data_val, scaling = data_val[1:], data_val[0, 1:]  # remove scaling row, remove round from scaling
    data_val_r, data_val_values = data_val[:, 0], data_val[:, 1:]  # seperate round from data

    data_val_max = data_val_values.max(axis=1)
    data_val_max_thresh = get_thresh(data_val_max)
    data_val_max_thresh = np.maximum(np.minimum(data_val_max_thresh, 1), 0)
    (w_smooth, exp_smooth) = get_smoothing(2 * data_val_max)

    plt.figure()
    plt.xlabel('Communication Round')
    plt.xlim(0, d_rounds)
    plt.ylim(-0.05, 1.1)

    plt.plot(data_val_r + 1, np.minimum(2 * data_val_max, 1), '-k')
    plt.plot(data_val_r + 1, np.minimum(w_smooth, 1), '--r')
    plt.plot(data_val_r + 1, np.minimum(exp_smooth, 1), '--', color='orange')
    plt.plot(data_val_r + 1, np.minimum(data_val_max_thresh, 1), '--b')

    plt.legend(labels=[
        'base threshold',
        'window=3 smoothing',
        'exponential smoothing',
        'global min-mean smoothing'
    ])

    if path is not None:
        plt.savefig(os.path.join(path, f'min_mean{suffix}--d_rounds{d_rounds}.png'), bbox_inches='tight')

    #plt.show()
    plt.close()


""" Main function HERE """
def main():

    # setup
    args = get_args()
    if args.d_rounds is None:
        args.d_rounds = args.n_rounds

    path = os.path.join(
        '/home/joe/03_federated/01_trusted/centralized',
        'alpha' + str(args.alpha) + '--alpha_val' + str(args.alpha_val)
    )
    suffix = '--alpha' + str(args.alpha) + '--alpha_val' + str(args.alpha_val)

    if not os.path.exists(os.path.join(path, 'visuals')):
        os.mkdir(path, 'visuals')


    """ Defense Plot - Thresholding """
    # import no attack nor defense data
    subdir = os.path.join(
        path,
        'n_rounds' + str(args.n_rounds) + '--d_start1--m_start1--n_malicious1'
    )

    temp_val = np.load(os.path.join(subdir, 'data/output_val_ks.npy'), allow_pickle=True)
    plot_threshold(temp_val, args.d_rounds, suffix)


if __name__ == "__main__":
    main()

