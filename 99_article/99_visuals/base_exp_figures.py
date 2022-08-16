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
    parser.add_argument('--alpha', default=10000, type=int)
    parser.add_argument('--alpha_val', default=10000, type=int)
    parser.add_argument('--m_start', default=1, type=int)

    return parser.parse_args()


""" Main function HERE """
def main():

    # setup
    args = get_args()

    methods = ['01_trusted', '00_baseline/mean', '00_baseline/median']
    titles = ['Trusted Threshold', 'Cordinate Median', 'Cordinate Trim Mean']

    paths = [
        os.path.join(
            '/home/joe/03_federated',
            method,
            ('distributed' if args.dba else 'centralized'),
            'alpha' + str(args.alpha) + '--alpha_val' + str(args.alpha_val)
        )
        for method in methods
    ]
    suffix = '--n_malicious' + str(args.n_malicious)

    if not os.path.exists(os.path.join(paths[0], 'visuals')):
        os.mkdir(os.path.join(path[0], 'visuals'))


    """ Global Accuracy """
    fig, axarr = plt.subplots(ncols=3, sharey=True, sharex=True, figsize=(12, 4))

    for i, p in enumerate(paths):
        f = 'data/output_global_acc.npy'

        # import files
        if os.path.exists(
            os.path.join(p, f'n_rounds{args.n_rounds}--m_start{args.m_start}--n_malicious{args.n_malicious}')
        ):

            temp_data = np.load(
                os.path.join(p, f'n_rounds{args.n_rounds}--m_start{args.m_start}--n_malicious{args.n_malicious}', f
                ), allow_pickle=True
            )

        else:
            temp_data = np.load(
                os.path.join(p, f'n_rounds{args.n_rounds}--d_start1--m_start{args.m_start}--n_malicious{args.n_malicious}', f
                ), allow_pickle=True
            )

        # global acc
        plt.sca(axarr[i])
        plt.title(titles[i])
        plt.xlabel('Communication Rounds')
        plt.plot(range(1, len(temp_data) + 1), temp_data[:, 1], '-b')
        plt.plot(range(1, len(temp_data) + 1), temp_data[:, 2], '-r')
        plt.ylim(-0.05, 1.1)

        plt.vlines(1, 0, 1, colors='r')
        plt.text(1, 1.033, 'attack start', c='r')

        copy = axarr[i].twinx()
        copy.set_ylim(-0.05, 1.1)
        copy.set_yticklabels([])

        if i == 0:
            axarr[0].set_ylabel('Classification Accuracy', c='b')

        if (i + 1) == len(titles):
            copy.set_ylabel('Attack Success Rate', c='r')

    plt.savefig(
        os.path.join(paths[0], 'visuals', f'baseline{suffix}.png'), bbox_inches='tight'
    )
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()

