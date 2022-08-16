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
    parser.add_argument('--dba', default=0, type=int)
    parser.add_argument('--n_rounds', default=250, type=int)
    parser.add_argument('--n_malicious', default=1, type=int)
    parser.add_argument('--m_start', default=1, type=int)
    parser.add_argument('--alpha', default=10000, type=int)
    parser.add_argument('--alpha_val', default=10000, type=int)

    return parser.parse_args()


""" Main function HERE """
def main():

    # setup
    args = get_args()
    if args.d_rounds is None:
        args.d_rounds = args.n_rounds

    methods = '01_trusted', '00_baseline/mean', '00_baseline/median'
    paths = [
        os.path.join(
            '/home/joe/03_federated',
            method,
            ('distributed' if args.dba else 'centralized'),
            'alpha' + str(args.alpha) + '--alpha_val' + str(args.alpha_val)
        )
        for method in methods
    ]
    suffix = '--n_malicious' + str(args.n_malicious) + '--m_start' + str(args.m_start)

    if not os.path.exists(os.path.join(path, 'visuals')):
        os.mkdir(path, 'visuals')


    """ Global Accuracy """
    fig, axarr = plt.subplots(ncols=3, sharey=True, sharex=True, figsize=(12, 4))

    for p in paths:

        # import files
        i, j = values[index]

        if j == args.m_start:

            subdir = os.path.join(
                path,
                'n_rounds' + str(args.n_rounds) + '--d_start' + str(i) + (
                    '--m_start' + str(j) + '--n_malicious' + str(args.n_malicious) if j == args.m_start else '--m_start' + str(j) + '--n_malicious' + str(1)
                )
            )

            temp_global = np.load(os.path.join(subdir, 'data/output_global_acc.npy'), allow_pickle=True)
            temp_val = np.load(os.path.join(subdir, 'data/output_val_ks.npy'), allow_pickle=True)
            temp_user = np.load(os.path.join(subdir, 'data/output_user_ks.npy'), allow_pickle=True)

        else:

            if i == args.n_rounds:
                to_reads = [f.path for f in os.scandir(path) if re.search('n_rounds250--d_start251--m_start251', f.path)]

            else:
                to_reads = [f.path for f in os.scandir(path) if re.search('n_rounds250--d_start1--m_start251', f.path)]

            temp_global = np.load(os.path.join(to_reads[0], 'data/output_global_acc.npy'), allow_pickle=True)
            temp_val = np.load(os.path.join(to_reads[0], 'data/output_val_ks.npy'), allow_pickle=True)
            temp_user = np.load(os.path.join(to_reads[0], 'data/output_user_ks.npy'), allow_pickle=True)

        # global acc
        plt.sca(axarr[index])
        plt.title(titles[index])
        plt.xlabel('Communication Rounds')
        plt.plot(range(1, len(temp_global) + 1), temp_global[:, 1], '-b')
        plt.plot(range(1, len(temp_global) + 1), temp_global[:, 2], '-r')
        plt.ylim(-0.05, 1.1)

        if j == args.m_start:
            plt.vlines(args.m_start, 0, 1, colors='r')
            plt.text(args.m_start, 1.033, 'attack start', c='r')

        copy = axarr[index].twinx()
        copy.set_ylim(-0.05, 1.1)
        copy.set_yticklabels([])

        if index == 0:
            axarr[0].set_ylabel('Classification Accuracy', c='b')

        if (index + 1) == len(titles):
            copy.set_ylabel('Attack Success Rate', c='r')

    plt.savefig(
        os.path.join(path, 'visuals', f'accuracy{suffix}.png'), bbox_inches='tight'
    )
    #plt.show()
    plt.close()


    """ Only attack plots """
    fig, axarr = plt.subplots(ncols=2, sharey=True, sharex=True, figsize=(8, 4))

    k = 0
    for index in range(len(titles)):

        # import files
        i, j = values[index]
        if j != args.m_start:
            continue

        subdir = os.path.join(
            path,
            'n_rounds' + str(args.n_rounds) + '--d_start' + str(i) + '--m_start' + str(j) + '--n_malicious' + str(args.n_malicious)
        )

        temp_global = np.load(os.path.join(subdir, 'data/output_global_acc.npy'), allow_pickle=True)
        temp_val = np.load(os.path.join(subdir, 'data/output_val_ks.npy'), allow_pickle=True)
        temp_user = np.load(os.path.join(subdir, 'data/output_user_ks.npy'), allow_pickle=True)

        # data manipulation

        # global acc
        plt.sca(axarr[k])
        plt.title(titles[index])
        plt.xlabel('Communication Rounds')
        plt.plot(range(1, len(temp_global) + 1), temp_global[:, 1], '-b')
        plt.plot(range(1, len(temp_global) + 1), temp_global[:, 2], '-r')
        plt.ylim(-0.05, 1.1)

        if j == args.m_start:
            plt.vlines(args.m_start, 0, 1, colors='r')
            plt.text(args.m_start, 1.033, 'attack start', c='r')

        copy = axarr[k].twinx()
        copy.set_ylim(-0.05, 1.1)
        copy.set_yticklabels([])

        if k == 0:
            axarr[0].set_ylabel('Classification Accuracy', c='b')
        else:
            copy.set_ylabel('Attack Success Rate', c='r')

        k += 1

    plt.savefig(
        os.path.join(path, 'visuals', f'attack_accuracy{suffix}.png'), bbox_inches='tight'
    )
    #plt.show()
    plt.close()








if __name__ == "__main__":
    main()

