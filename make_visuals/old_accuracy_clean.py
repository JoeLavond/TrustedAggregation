""" Packages """
# base
import argparse
import numpy as np
import os
from pathlib import Path
import re

# visual
import matplotlib
import matplotlib.pyplot as plt
from  matplotlib.lines import Line2D
import seaborn as sns


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='cifar', type=str)
    parser.add_argument('--n_classes', default=10, type=int)

    parser.add_argument('--alpha', default=10000, type=int)
    parser.add_argument('--alpha_val', default=10000, type=int)

    parser.add_argument('--n_rounds', default=50, type=int)
    parser.add_argument('--m_start', default=1, type=int)

    parser.add_argument('--n_malicious', default=1, type=int)
    parser.add_argument('--dba', default=0, type=int)

    parser.add_argument('--neuro', default=0, type=int)
    parser.add_argument('--neuro_p', default=0.1, type=float)

    parser.add_argument('--beta', default=0.2, type=float)

    parser.add_argument('--d_rounds', default=None, type=int)
    parser.add_argument('--show', default=1, type=int)

    return parser.parse_args()


""" Main function HERE """
def main():

    # setup
    args = get_args()
    if args.d_rounds is None:
        args.d_rounds = args.n_rounds

    # source paths for our method
    tag_path = os.path.join(
        f'{Path.home()}/fed-learn-dba/{args.data}_{args.n_classes}',
        ('neuro' if args.neuro else 'classic'),
        'tag',
        ('distributed' if args.dba else 'centralized'),
        'alpha' + str(args.alpha) + '--alpha_val' + str(args.alpha_val)
    )
    centralized_path = os.path.join(
        f'{Path.home()}/fed-learn-dba/{args.data}_{args.n_classes}',
        ('neuro' if args.neuro else 'classic'),
        'tag/centralized',
        'alpha' + str(args.alpha) + '--alpha_val' + str(args.alpha_val)
    )
    suffix = f'--n_malicious{args.n_malicious}--beta{args.beta}'  # f'--m_start{args.m_start}''

    # control varibles for baseline methods
    methods = ['base/median', 'base/mean']
    base_paths = [
        os.path.join(
            f'{Path.home()}/fed-learn-dba/{args.data}_{args.n_classes}',
            ('neuro' if args.neuro else 'classic'),
            method,
            ('distributed' if args.dba else 'centralized'),
            'alpha' + str(args.alpha) + '--alpha_val' + str(args.alpha_val)
        ) for method in methods
    ]
    line_types = ['dotted', 'dashed']
    line_colors = ['blue', 'green']

    custom_lines = [
        Line2D([0], [0], linestyle='-', color='black', label='Tag'),
        Line2D([0], [0], linestyle='dotted', color='blue', label='Median'),
        Line2D([0], [0], linestyle='dashed', color='green', label=f'Trim Mean (Beta={args.beta})')
    ]

    if not os.path.exists(os.path.join(tag_path, 'visuals')):
        os.makedirs(os.path.join(tag_path, 'visuals'))

    # experiments
    titles = ['No Attack or Defense', 'Attack Only', 'Defense Only', 'Attack and Defense']
    values = [
        (args.n_rounds + 1, args.n_rounds + 1),  # no attack or defense
        (args.n_rounds + 1, args.m_start),  # attack only
        (1, args.n_rounds + 1),  # defense only
        (1, args.m_start)  # attack and defense
   ]


    """ Global Accuracy """
    fig, axarr = plt.subplots(ncols=4, sharey=True, sharex=True, figsize=(16, 4))

    for index in range(len(titles)):
        i, j = values[index]

        """ Experiment data """
        # get data to plot
        if j == args.m_start:  # attack is present
            subdir = os.path.join(
                tag_path,
                f'n_rounds{args.n_rounds}--d_start{i}--m_start{j}--n_malicious{args.n_malicious}'
            )
        else:  # no attack is present
            subdir = os.path.join(
                centralized_path,
                f'n_rounds{args.n_rounds}--d_start{i}--m_start{args.n_rounds + 1}--n_malicious1'
            )

        temp_global = np.load(
            os.path.join(
                subdir,
                (
                    'data/output_global_acc'
                    + (f'--neuro_p{args.neuro_p}' if args.neuro else '')
                    + '.npy'
                )
            ), allow_pickle=True
        )

        # plot data from experiments
        plt.sca(axarr[index])
        plt.title(titles[index])
        plt.xlabel('Communication Round')
        plt.plot(range(0, args.d_rounds + 1), temp_global[:args.d_rounds + 1, 1], '-k')


        """ Baseline data """
        if i == 1 and j == args.m_start:  # only compare base under attack and defense
            for p, lt, col in zip(base_paths, line_types, line_colors):  # iterate baseline defenses

                # import baseline data
                subdir = [
                    f.path for f in os.scandir(p) if re.search(f'm_start{args.m_start}', f.path)
                ]
                temp_global = np.load(
                    os.path.join(
                        subdir[0],
                        (
                            'data/output_global_acc'
                            + (f'--neuro_p{args.neuro_p}' if args.neuro else '')
                            + (f'--beta{args.beta}' if re.search('mean', p) else '')
                            + '.npy'
                        )
                    ),
                    allow_pickle=True
                )

                plt.plot(range(0, args.d_rounds + 1), temp_global[:args.d_rounds + 1, 1], c=col, linestyle=lt)

            plt.legend(handles=custom_lines, loc=0)

        # plot axis info
        if index == 0:
            axarr[0].set_ylabel('Classification Accuracy')

    plt.savefig(
        os.path.join(tag_path, 'visuals', f'clean_accuracy{suffix}.png'), bbox_inches='tight'
    )
    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()

