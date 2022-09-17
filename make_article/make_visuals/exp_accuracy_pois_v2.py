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

    parser.add_argument('--alpha', default=10000, type=int)
    parser.add_argument('--alpha_val', default=10000, type=int)

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

    args = get_args()
    out_suffix = (
        f'--n_malicious{args.n_malicious}--dba{args.dba}--beta{args.beta}'
        + (f'--neuro_p{args.neuro_p}' if args.neuro else '')
        + '--d_rounds{d_rounds}'
    )

    # hyper-parameters
    datasets = ('cifar_10', 'cifar_100', 'stl_10')
    n_rounds = (50, 50, 100)
    subdirs = [
        f'--d_start1--m_start{args.m_start}--n_malicious{args.n_malicious}',
        f'--m_start{args.m_start}--n_malicious{args.n_malicious}',
        f'--m_start{args.m_start}--n_malicious{args.n_malicious}'
    ]

    methods = ('tag', 'base/mean', 'base/median')
    file_suffices = ('', f'--beta{args.beta}', '')

    line_types = ['solid', 'dotted', 'dashed']
    warm_colors = ['yellow', 'orange', 'red']
    cool_colors = ['green', 'blue', 'purple']

    # setup
    if args.d_rounds is None:
        args.d_rounds = min(n_rounds)

    out_path = './visuals'
    if not os.path.exists('./visuals'):
        os.makedirs('./visuals')

    custom_lines = (
        [
            Line2D([0], [0], linestyle=ls, color=lc, label=method)
            for ls, lc, method in zip(line_types, cool_colors, methods)
        ] + [
            Line2D([0], [0], linestyle=ls, color=lc, label=method)
            for ls, lc, method in zip(line_types, warm_colors, methods)
        ]
    )


    """ Global Accuracy """
    fig, axarr = plt.subplots(ncols=3, sharey=True, sharex=True, figsize=(12, 4))

    for i, data in enumerate(datasets):
        plt.sca(axarr[i])
        plt.title(data)
        plt.xlabel('Communication Round')

        if i == 0:
            plt.ylabel('Classification Rate', c=cool_colors[i])
        elif i == len(datasets):
            plt.ylabel('Attack Success Rate', c=warm_colors[i])

        for j, method in enumerate(methods):

            path = os.path.join(
                f'{Path.home()}/fed-learn-dba',
                data,
                ('neuro' if args.neuro else 'classic'),
                method,
                ('distributed' if args.dba else 'centralized'),
                'alpha' + str(args.alpha) + '--alpha_val' + str(args.alpha_val)
            )

            temp_global = np.load(
                os.path.join(
                    path,
                    (
                        f'n_rounds{n_rounds[i]}'
                        + subdirs[j]
                    ),
                    'data',
                    (
                        'output_global_acc'
                        + ('--neuro_p{args.neuro_p}' if args.neuro else '')
                        + file_suffices[j]
                        + '.npy'
                    )
                ), allow_pickle=True
            )

            plt.plot(range(0, args.d_rounds + 1), temp_global[:args.d_rounds + 1, 1], c=cool_colors[j])  # need to specify line type
            plt.plot(range(0, args.d_rounds + 1), temp_global[:args.d_rounds + 1, 2], c=warm_colors[j])


    plt.savefig(
        os.path.join('./visuals', f'pois_accuracy{out_suffix}.png'),
        bbox_inches='tight'
    )
    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()

