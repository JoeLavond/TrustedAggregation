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

    parser.add_argument('--resnet', default=1, type=int)
    parser.add_argument('--alpha', default=10000, type=int)
    parser.add_argument('--alpha_val', default=10000, type=int)
    parser.add_argument('--n_val_data', default=None, type=int)

    parser.add_argument('--n_rounds', default=250, type=int)
    parser.add_argument('--d_rounds', default=None, type=int)
    parser.add_argument('--m_start', default=1, type=int)
    parser.add_argument('--n_malicious', default=1, type=int)
    parser.add_argument('--dba', default=0, type=int)

    parser.add_argument('--neuro', default=0, type=int)
    parser.add_argument('--neuro_p', default=0.1, type=float)

    parser.add_argument('--beta', default=0.2, type=float)
    parser.add_argument('--d_scale', default=2, type=float)
    parser.add_argument('--d_smooth', default=1, type=float)

    parser.add_argument('--font_size', default=14, type=int)
    parser.add_argument('--show', default=1, type=int)

    return parser.parse_args()


""" Main function HERE """
def main():

    args = get_args()
    out_suffix = (
        f'--n_malicious{args.n_malicious}--dba{args.dba}--beta{args.beta}'
        + (f'--d_scale{args.d_scale}' if args.d_scale != 2. else '')
        + (f'--neuro_p{args.neuro_p}' if args.neuro else '')
        + (f'--n_val_data{args.n_val_data}' if args.n_val_data is not None else '')
        + (f'--alpha{args.alpha}' if args.alpha != 10000 else '')
        + ('--no_smooth' if not args.d_smooth else '')
        + ('--vgg' if not args.resnet else '')
    )

    """ Hyperparams """
    font = {
        'size': args.font_size
    }
    mpl.rc('font', **font)

    data = ['cifar_10', 'cifar_100', 'stl_10']
    out_data = ['CIFAR-10', 'CIFAR-100', 'STL-10']

    methods = ('tag', 'base/mean', 'base/median', 'base/trust')
    subdirs = [
        f'--d_start1--m_start{args.m_start}--n_malicious{args.n_malicious}',
        f'--m_start{args.m_start}--n_malicious{args.n_malicious}',
        f'--m_start{args.m_start}--n_malicious{args.n_malicious}',
        f'--m_start{args.m_start}--n_malicious{args.n_malicious}'
    ]

    d_scale = [None, None, 1.1]
    args.d_rounds = args.n_rounds if args.d_rounds is None else args.d_rounds

    line_styles = ['solid', 'dotted', 'dashed', 'dashdot']
    cool_colors = ['#88CCEE', '#44AA99', '#117733', '#999933']
    warm_colors = ['#EE7733', '#CC6677', '#CC3311', '#AA4499']

    out_path = './experiment_figures'
    if not os.path.exists('./experiment_figures'):
        os.makedirs('./experiment_figures')


    """ Global Accuracy """
    fig, axarr = plt.subplots(ncols=len(data), figsize=(len(data)*4, 4))
    for i, (d, out_d) in enumerate(zip(data, out_data)):
        plt.sca(axarr[i])
        plt.title(out_d, fontsize=1.5*args.font_size)
        plt.xlabel('Communication Round')

        clean_lines = []
        pois_lines = []

        file_suffices = (
            (
                f'--d_scale{d_scale[i]}'
                if d_scale[i] is not None
                else ''
                + (
                    f'--n_val_data{args.n_val_data}'
                    if args.n_val_data is not None
                    else ''
                )
                + (
                    '--no_smooth'
                    if not args.d_smooth
                    else ''
                )
            ),
            f'--beta{args.beta}',
            '',
            ''
        )

        for j, method in enumerate(methods):

            path = os.path.join(
                f'{Path.home()}/TrustedAggregation',
                d,
                ('neuro' if args.neuro else 'classic'),
                method,
                ('distributed' if args.dba else 'centralized'),
                'alpha' + str(args.alpha) + '--alpha_val' + str(args.alpha_val)
            )

            temp_global = np.load(
                os.path.join(
                    path,
                    (
                        f'n_rounds{args.n_rounds}'
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

            clean_line, = plt.plot(range(0, args.d_rounds + 1), temp_global[:args.d_rounds + 1, 1], c=cool_colors[j], linestyle=line_styles[j], label=method)
            clean_lines.append(clean_line)
            pois_line, = plt.plot(range(0, args.d_rounds + 1), temp_global[:args.d_rounds + 1, 2], c=warm_colors[j], linestyle=line_styles[j], label=method)
            pois_lines.append(pois_line)

    # create legend
    out_methods = ['Trusted Aggregation', 'Coordinate Median', 'Coordinate Trim-Mean', 'FLTrust']
    l1 = plt.legend(
        clean_lines,
        out_methods,
        title='Classification Accuracy',
        bbox_to_anchor=(1.04, 1.04),
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
        bbox_to_anchor=(1.04, -0.04),
        loc='lower left'
    )

    for i, c in enumerate(warm_colors):
        l2.legendHandles[i].set_color(c)
        l2.legendHandles[i].set_linestyle(line_styles[i])

    plt.savefig(
        os.path.join(out_path, f'accuracy{out_suffix}.png'),
        bbox_inches='tight'
    )

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()

