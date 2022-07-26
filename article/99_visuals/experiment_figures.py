""" Packages """
# base
import argparse
import numpy as np
import os

# visual
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
#matplotlib.pyplot.switch_backend('agg')
import seaborn as sns


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dba', default=0, type=int)
    parser.add_argument('--n_rounds', default=250, type=int)
    parser.add_argument('--d_rounds', default=None, type=int)
    parser.add_argument('--n_malicious', default=1, type=int)
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


def plot_scaling(data_val, data_user, d_rounds, path):

    # validation user
    data_val, data_user = np.array(data_val), np.array(data_user)
    data_val = data_val[:-1]  # remove last validation check for next iteration never run
    data_val, scaling = data_val[1:], data_val[0, 1:]  # remove scaling row, remove round from scaling
    print('scaling:', scaling)
    data_val_r, data_val_values = data_val[:, 0], data_val[:, 1:]  # seperate round from data

    data_val_max = data_val_values.max(axis=1)
    data_val_max_thresh = get_thresh(data_val_max)
    data_val_scaled = data_val_values * scaling
    data_val_scaled_max = data_val_scaled.max(axis=1)
    data_val_scaled_max_thresh = get_thresh(data_val_scaled_max)

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
    plt.xlim(1, d_rounds)

    plt.title('Entropy Scaled Threshold Over Rounds')
    plt.plot(data_val_r, benign_upper, '--b')
    plt.plot(data_val_r, benign_lower, '--b')
    plt.plot(data_val_r, data_val_max_thresh, '-r')
    plt.plot(data_val_r, data_val_scaled_max_thresh, '-g')
    plt.legend(labels=[
        'benign max ks: q3 + 1.5 * IQR',
        'benign max ks: q1 - 1.5 * IQR',
        'unscaled threshold',
        'scaled threshold'
    ])

    if path is not None:
        plt.savefig(os.path.join(path, 'visuals', f'scaling--d_rounds{d_rounds}.png'))

    #plt.show()
    plt.close()


def plot_threshold(data_val, data_user, d_rounds, path, n_malicious):

    # validation user
    data_val, data_user = np.array(data_val), np.array(data_user)
    data_val = data_val[:-1]  # remove last validation check for next iteration never run
    data_val, scaling = data_val[1:], data_val[0, 1:]  # remove scaling row, remove round from scaling
    print('scaling:', scaling)
    data_val_r, data_val_values = data_val[:, 0], data_val[:, 1:]  # seperate round from data

    data_val_max = data_val_values.max(axis=1)
    data_val_max_thresh = get_thresh(data_val_max)
    data_val_scaled = data_val_values * scaling
    data_val_scaled_max = data_val_scaled.max(axis=1)
    data_val_scaled_max_thresh = get_thresh(data_val_scaled_max)

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

    # malicious users
    data_malicious = data_user[data_user[:, 0] == 1, 1:]  # subset to benign users, remove malicious index column
    data_malicious_r, data_malicious_values = data_malicious[:, 0], data_malicious[:, 1:]  # seperate round column from data
    data_malicious_max = data_malicious_values.max(axis=1)

    fig1, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(8, 4))

    # visual - benign
    ax1.set_title('Threshold vs. Benign Over Rounds')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlabel('Communication Round')
    ax1.set_xlim(1, d_rounds)

    ax1.plot(data_val_r, benign_upper, '--b')
    ax1.plot(data_val_r, benign_lower, '--b')
    ax1.plot(data_val_r, data_val_scaled_max_thresh, '-g')
    ax1.legend(labels=[
        'benign max ks: q3 + 1.5 * IQR',
        'benign max ks: q1 - 1.5 * IQR',
        'scaled threshold'
    ])

    # visual - malicious
    ax2.set_title('Threshold vs. Malicious Over Rounds')
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel('Communication Round')
    ax2.set_xlim(1, d_rounds)

    if n_malicious > 1:


        # get lower and upper bound for malicious users
        data_malicious_max_q1, data_malicious_max_q3 = get_quantiles(data_malicious_r, data_malicious_max)
        malicious_upper = data_malicious_max_q3 + 1.5 * (data_malicious_max_q3 - data_malicious_max_q1)
        malicious_upper = np.maximum(np.minimum(malicious_upper, 1), 0)
        malicious_lower = data_malicious_max_q1 - 1.5 * (data_malicious_max_q3 - data_malicious_max_q1)
        malicious_lower = np.maximum(np.minimum(malicious_lower, 1), 0)

        ax2.plot(data_val_r, malicious_upper, '--r')
        ax2.plot(data_val_r, malicious_lower, '--r')
        ax2.plot(data_val_r, data_val_scaled_max_thresh, '-g')

        c_labels=[
            'malicious max ks: q3 + 1.5 * IQR',
            'malicious max ks: q1 - 1.5 * IQR',
            'scaled threshold'
        ]
        ax2.legend(labels=c_labels)

    else:

        ax2.plot(data_val_r, data_malicious_max, '--r')
        ax2.plot(data_val_r, data_val_scaled_max_thresh, '-g')

        c_labels=[
            'malicious max ks',
            'scaled threshold'
        ]
        ax2.legend(labels=c_labels)

    if path is not None:
        plt.savefig(os.path.join(path, 'visuals', f'threshold--d_rounds{d_rounds}.png'))

    #plt.show()
    plt.close()


""" Main function HERE """
def main():

    # setup
    args = get_args()
    if args.d_rounds is None:
        args.d_rounds = args.n_rounds

    path = os.path.join(
        '/home/joe/03_federated',
        ('distributed' if args.dba else 'centralized'),
        'alpha' + str(args.alpha) + '--alpha_val' + str(args.alpha_val)
    )

    if not os.path.exists(os.path.join(path, 'visuals')):
        os.mkdir(os.path.join(path, 'visuals'))

    # experiments
    titles = ['No Attack or Defense', 'Attack Only', 'Defense Only', 'Attack and Defense']
    values = [
        (args.n_rounds + 1, args.n_rounds + 1),
        (args.n_rounds + 1, 1),
        (1, args.n_rounds + 1),
        (1, 1)
   ]


    """ Defense Plot - Entropy Scaling """
    # import no attack nor defense data
    (i, j) = values[0]
    subdir = os.path.join(
        path,
        'n_rounds' + str(args.n_rounds) + '--d_start' + str(i) + '--m_start' + str(j) + '--n_malicious' + str(args.n_malicious)
    )

    temp_val = np.load(os.path.join(subdir, 'data/output_val_ks.npy'), allow_pickle=True)
    temp_user = np.load(os.path.join(subdir, 'data/output_user_ks.npy'), allow_pickle=True)

    plot_scaling(temp_val, temp_user, args.d_rounds, path)


    """ Defense Plot - Thresholding """
    # import no attack nor defense data
    (i, j) = values[-1]
    subdir = os.path.join(
        path,
        'n_rounds' + str(args.n_rounds) + '--d_start' + str(i) + '--m_start' + str(j) + '--n_malicious' + str(args.n_malicious)
    )

    temp_val = np.load(os.path.join(subdir, 'data/output_val_ks.npy'), allow_pickle=True)
    temp_user = np.load(os.path.join(subdir, 'data/output_user_ks.npy'), allow_pickle=True)

    plot_threshold(temp_val, temp_user, args.d_rounds, path, args.n_malicious)


    """ Global Accuracy """
    fig, axarr = plt.subplots(ncols=4, sharey=True, figsize=(16, 4))
    for index in range(len(titles)):

        # import files
        i, j = values[index]
        subdir = os.path.join(
            path,
            'n_rounds' + str(args.n_rounds) + '--d_start' + str(i) + '--m_start' + str(j) + '--n_malicious' + str(args.n_malicious)
        )

        temp_global = np.load(os.path.join(subdir, 'data/output_global_acc.npy'), allow_pickle=True)
        temp_val = np.load(os.path.join(subdir, 'data/output_val_ks.npy'), allow_pickle=True)
        temp_user = np.load(os.path.join(subdir, 'data/output_user_ks.npy'), allow_pickle=True)

        # data manipulation

        # global acc
        plt.sca(axarr[index])
        plt.title(titles[index])
        plt.plot(range(1, len(temp_global) + 1), temp_global[:, 1], '-b')
        plt.plot(range(1, len(temp_global) + 1), temp_global[:, 2], '-r')
        plt.ylim(-0.05, 1.05)
        plt.xlabel('Communication Round')


    fig.legend(labels=['Accuracy', 'Attack Success Rate'], loc=7)
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)

    plt.savefig(
        os.path.join(path, 'visuals', 'accuracy.png')
    )
    #plt.show()
    plt.close()








if __name__ == "__main__":
    main()

