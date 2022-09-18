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
    )

    # hyper-parameters
    datasets = ('cifar_10', 'cifar_100', 'stl_10')
    if args.neuro:
        n_rounds = (250, 250, 100)
    else:
        n_rounds = (250, 250, 100)

    d_rounds = n_rounds
    subdirs = [
        f'--d_start1--m_start{args.m_start}--n_malicious{args.n_malicious}',
        f'--m_start{args.m_start}--n_malicious{args.n_malicious}',
        f'--m_start{args.m_start}--n_malicious{args.n_malicious}'
    ]

    methods = ('tag', 'base/mean', 'base/median')
    file_suffices = ('', f'--beta{args.beta}', '')

    line_styles = ['solid', 'dotted', 'dashed']
    warm_colors = [
        (238 / 255, 51 / 255, 119 / 255),
        (238 / 255, 119 / 255, 51 / 255),
        (204 / 255, 51 / 255, 17 / 255)
    ]
    cool_colors = [
        (0 / 255, 153 / 255, 136 / 255),
        (0 / 255, 119 / 255, 187 / 255),
        (51 / 255, 187 / 255, 238 / 255)
    ]

    out_path = './visuals'
    if not os.path.exists('./visuals'):
        os.makedirs('./visuals')


    """ Global Accuracy """
    fig, axarr = plt.subplots(ncols=3, figsize=(12, 4))

    for i, data in enumerate(datasets):
        plt.sca(axarr[i])
        plt.title(data)
        plt.xlabel('Communication Round')

        clean_lines = []
        pois_lines = []

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
                        + (f'--neuro_p{args.neuro_p}' if args.neuro else '')
                        + file_suffices[j]
                        + '.npy'
                    )
                ), allow_pickle=True
            )

            clean_line, = plt.plot(range(0, d_rounds[i] + 1), temp_global[:d_rounds[i] + 1, 1], c=cool_colors[j], linestyle=line_styles[j], label=method)
            clean_lines.append(clean_line)
            pois_line, = plt.plot(range(0, d_rounds[i] + 1), temp_global[:d_rounds[i] + 1, 2], c=warm_colors[j], linestyle=line_styles[j], label=method)
            pois_lines.append(pois_line)

    # create legend
    l1 = plt.legend(
        clean_lines,
        title='classification accuracy',
        bbox_to_anchor=(1.04, 1),
        loc='upper left'
    )
    ax = plt.gca().add_artist(l1)

    for i, c in enumerate(cool_colors):
        l1.legendHandles[i].set_color(c)

    l2 = plt.legend(
        pois_lines,
        title='attack success rate',
        bbox_to_anchor=(1.04, 0),
        loc='lower left'
    )

    for i, c in enumerate(warm_colors):
        l2.legendHandles[i].set_color(c)

    plt.savefig(
        os.path.join('./visuals', f'accuracy{out_suffix}.png'),
        bbox_inches='tight'
    )
    if args.show:
        plt.tight_layout()
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()

