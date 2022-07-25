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
    parser.add_argument('--n_rounds', default=10, type=int)
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


""" Main function HERE """
def main():

    args = get_args()
    path = os.path.join(
        '/home/joe/03_federated',
        ('distributed' if args.dba else 'centralized'),
        'alpha' + str(args.alpha) + '--alpha_val' + str(args.alpha_val)
    )

    if not os.path.exists(os.path.join(path, 'visuals')):
        os.mkdir(os.path.join(path, 'visuals'))


    """ Global Accuracy """
    fig, axarr = plt.subplots(ncols=4, sharey=True, figsize=(16, 4))

    titles = ['No Attack or Defense', 'Attack Only', 'Defense Only', 'Attack and Defense']
    values = [
        (args.n_rounds + 1, args.n_rounds + 1),
        (args.n_rounds + 1, 1),
        (1, args.n_rounds + 1),
        (1, 1)
   ]


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

        if i == args.n_rounds + 1 and j == args.n_rounds + 1:
            data_global = temp_global
            data_val = temp_val
            data_user = temp_user

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
        os.path.join(path, 'visuals', 'global_acc')
    )
    #plt.show()
    plt.close()


    """ Defense Plots """
    # benign users
    data_val, data_user = np.array(data_val), np.array(data_user)
    data_benign = data_user[data_user[:, 0] == 0, 1:]
    data_val = data_val[:-1]
    data_benign_r, data_benign_values = data_benign[:, 0], data_benign[:, 1:]

    data_val, scaling = data_val[1:], data_val[0, 1:]
    data_val_r, data_val_values = data_val[:, 0], data_val[:, 1:]

    data_benign_max = data_benign_values.max(axis=1)
    data_benign_max_q1, data_benign_max_q3 = get_quantiles(data_benign_r, data_benign_max)

    data_val_max = data_val_values.max(axis=1)
    data_val_max_thresh = get_thresh(data_val_max)
    data_val_scaled = data_val_values / scaling
    data_val_scaled_max = data_val_scaled.max(axis=1)
    data_val_scaled_max_thresh = get_thresh(data_val_scaled_max)

    user_upper = data_benign_max_q3 + 1.5 * (data_benign_max_q3 - data_benign_max_q1),
    user_upper = np.maximum(np.minimum(user_upper, 1), 0)
    user_lower = data_benign_max_q1 - 1.5 * (data_benign_max_q3 - data_benign_max_q1),
    user_lower = np.maximum(np.minimum(user_lower, 1), 0)

    data_val_max_thresh = np.maximum(np.minimum(data_val_max_thresh, 1), 0)
    data_val_scaled_max_thresh = np.maximum(np.minimum(data_val_scaled_max_thresh, 1), 0)

    plt.figure()
    plt.title('Entropy Scaled Threshold Over Rounds')
    plt.plot(data_val_r, user_upper, '--b')
    plt.plot(data_val_r, user_lower, '--b')
    plt.plot(data_val_r, data_val_max_thresh, '-r')
    plt.plot(data_val_r, data_val_scaled_max_thresh, '-g')
    plt.legend(labels=[
        'benign users: q3 + 1.5 * IQR',
        'benign users: q1 - 1.5 * IQR',
        'val user: unscaled',
        'val user: scaled'
    ])
    plt.show()
    plt.close()






if __name__ == "__main__":
    main()

