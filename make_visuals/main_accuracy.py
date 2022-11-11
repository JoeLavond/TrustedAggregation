""" Packages """
# base
import argparse
import numpy as np
import os
from pathlib import Path
import re

# visual
import matplotlib as mpl
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
    parser.add_argument('--d_scale', default=1, type=int)

    parser.add_argument('--d_rounds', default=None, type=int)
    parser.add_argument('--font_size', default=14, type=int)
    parser.add_argument('--show', default=1, type=int)

    return parser.parse_args()


""" Main function HERE """
def main():

    args = get_args()
    # hyper-parameters
    font = {
        'size': args.font_size
    }
    mpl.rc('font', **font)

    datasets = ('cifar_10', 'cifar_100', 'stl_10')
    if args.neuro:
        n_rounds = (250, 250, 250)
    else:
        n_rounds = (250, 250, 250)

    if args.d_scale:
        d_scale = (None, None, 1.1)
    else:
        d_scale = (None, None, None)

    out_suffix = (
        f'--n_malicious{args.n_malicious}--dba{args.dba}--beta{args.beta}'
        + ('--d_scale' if args.d_scale else '')
        + (f'--neuro_p{args.neuro_p}' if args.neuro else '')
    )

    d_rounds = n_rounds
    subdirs = [
        f'--d_start1--m_start{args.m_start}--n_malicious{args.n_malicious}',
        f'--m_start{args.m_start}--n_malicious{args.n_malicious}',
        f'--m_start{args.m_start}--n_malicious{args.n_malicious}'
    ]

    methods = ('tag', 'base/mean', 'base/median')
    file_suffices = (
        '',
        f'--beta{args.beta}',
        ''
    )

    line_styles = ['solid', 'dotted', 'dashed']
    warm_colors = ['pink', 'red', 'orange']
    cool_colors = ['blue', 'cyan', 'green']

    """
    # scale RGB between 0-1
    for i, (wc, cc) in enumerate(zip(warm_colors, cool_colors)):
        warm_colors[i] = [x / 255 for x in wc]
        cool_colors[i] = [x / 255 for x in cc]

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
    """

    out_path = './visuals'
    if not os.path.exists('./visuals'):
        os.makedirs('./visuals')


    """ Global Accuracy """
    out_datasets = ['CIFAR-10', 'CIFAR-100', 'STL-10']
    fig, axarr = plt.subplots(ncols=3, figsize=(12, 4))

    for i, data in enumerate(datasets):
        plt.sca(axarr[i])
        plt.title(out_datasets[i], fontsize=1.5*args.font_size)
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
                        + (
                            f'--d_scale{d_scale[i]}'
                            if (d_scale[i] is not None and method == 'tag')
                            else ''
                        )
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
    out_methods = ['Trusted Aggregation', 'Coordinate Median', 'Coordinate Trim-Mean']
    l1 = plt.legend(
        clean_lines,
        out_methods,
        title='Classification Accuracy',
        bbox_to_anchor=(1.04, 1),
        loc='upper left'
    )
    ax = plt.gca().add_artist(l1)

    for i, c in enumerate(cool_colors):
        l1.legendHandles[i].set_color(c)
        l1.legendHandles[i].set_linestyle(line_styles[i])

    l2 = plt.legend(
        pois_lines,
        out_methods,
        title='Attack Success Rate',
        bbox_to_anchor=(1.04, 0),
        loc='lower left'
    )

    for i, c in enumerate(warm_colors):
        l2.legendHandles[i].set_color(c)
        l2.legendHandles[i].set_linestyle(line_styles[i])

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
