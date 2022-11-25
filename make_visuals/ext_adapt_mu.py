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

    parser.add_argument('--data', default='cifar', type=str)
    parser.add_argument('--n_classes', default=10, type=int)
    parser.add_argument('--resnet', default=1, type=int)

    parser.add_argument('--alpha', default=10000, type=int)
    parser.add_argument('--alpha_val', default=10000, type=int)
    parser.add_argument('--n_val_data', default=None, type=int)

    parser.add_argument('--m_start', default=1, type=int)
    parser.add_argument('--n_malicious', default=1, type=int)
    parser.add_argument('--dba', default=0, type=int)

    parser.add_argument('--neuro', default=0, type=int)
    parser.add_argument('--neuro_p', default=0.1, type=float)

    parser.add_argument('--beta', default=0.2, type=float)
    parser.add_argument('--mu', default=0, type=float)
    parser.add_argument('--d_scale', default=2, type=float)
    parser.add_argument('--d_smooth', default=1, type=float)

    parser.add_argument('--n_rounds', default=250, type=int)
    parser.add_argument('--d_rounds', default=None, type=int)

    parser.add_argument('--font_size', default=14, type=int)
    parser.add_argument('--show', default=1, type=int)

    return parser.parse_args()


""" Main function HERE """
def main():

    args = get_args()
    out_suffix = (
        f'--n_malicious{args.n_malicious}--dba{args.dba}--beta{args.beta}'
        + (f'--neuro_p{args.neuro_p}' if args.neuro else '')
        + (f'--mu{args.mu}' if args.mu > 0. else '')
    )

    # hyper-parameters
    font = {
        'size': args.font_size
    }
    mpl.rc('font', **font)

    data = f'{args.data}_{args.n_classes}'
    out_data = f'{args.data.upper()}-{args.n_classes}'

    if args.d_rounds is None:
        args.d_rounds = args.n_rounds

    subdir = f'--d_start1--m_start{args.m_start}--n_malicious{args.n_malicious}'

    d_scales = [1.1, 1.25, 1.5, 2.0]
    file_suffices = [
        (
            (
                f'--d_scale{d_scale}'
                if d_scale != 2.
                else ''
            )
            + (
                f'--mu{args.mu}'
                if args.mu > 0
                else ''
            )
        )
        for d_scale in d_scales
    ]

    line_styles = ['solid', 'dotted', 'dashed', 'dashdot']
    cool_colors = sns.color_palette("bright").as_hex()
    cool_colors = cool_colors[:len(file_suffices)]
    warm_colors = sns.color_palette("dark").as_hex()
    warm_colors = warm_colors[:len(file_suffices)]

    out_path = './extra'
    if not os.path.exists('./extra'):
        os.makedirs('./extra')


    """ Global Accuracy """
    plt.figure(figsize=(4, 4))

    plt.title(out_data, fontsize=1.5*args.font_size)
    plt.xlabel('Communication Round')

    to_remove = []

    for j, file_suffix in enumerate(file_suffices):

        path = os.path.join(
            f'{Path.home()}/fed-tag',
            data,
            ('neuro' if args.neuro else 'classic'),
            'adapt',
            ('distributed' if args.dba else 'centralized'),
            'alpha' + str(args.alpha) + '--alpha_val' + str(args.alpha_val)
        )

        f_path = os.path.join(
            (
                f'n_rounds{args.n_rounds}'
                + subdir
            ),
            'data',
            (
                'output_global_acc'
                + (f'--neuro_p{args.neuro_p}' if args.neuro else '')
                + file_suffix
                + '.npy'
            )
        )
        print(os.path.join(path, f_path))

        try:
            temp_global = np.load(
                os.path.join(
                    path, f_path
                ), allow_pickle=True
            )

            plt.plot(range(0, args.d_rounds + 1), temp_global[:args.d_rounds + 1, 1], c=cool_colors[j], linestyle=line_styles[j])
            plt.plot(range(0, args.d_rounds + 1), temp_global[:args.d_rounds + 1, 2], c=warm_colors[j], linestyle=line_styles[j])

        except:
            to_remove.append(j)

    # create legend
    l1 = plt.legend(
        [
            Line2D([0], [0])
            for i,  _ in enumerate(d_scales)
            if i not in to_remove
        ],
        [
            f'dscale{d_scale}'
            for i, d_scale in enumerate(d_scales)
            if i not in to_remove
        ],
        title="Classification Accuracy",
        bbox_to_anchor=(1.05, 1.05),
        loc='upper left'
    )
    ax = plt.gca().add_artist(l1)

    for i, h in enumerate(l1.legendHandles):
        h.set_color(cool_colors[i % len(cool_colors)])
        h.set_linestyle(line_styles[i % len(line_styles)])

    l2 = plt.legend(
        [
            Line2D([0], [0])
            for _ in d_scales
            if i not in to_remove
        ],
        [
            f'dscale{d_scale}'
            for d_scale in d_scales
            if i not in to_remove
        ],
        title="Attack Success Rate",
        bbox_to_anchor=(1.05, -0.05),
        loc='lower left'
    )
    ax = plt.gca().add_artist(l2)

    plot_ind = [
        i for i in range(len(d_scales))
        if i not in to_remove
    ]

    for i, h in enumerate(l2.legendHandles):
        h.set_color(warm_colors[plot_ind[i] % len(warm_colors)])
        h.set_linestyle(line_styles[plot_ind[i] % len(line_styles)])


    plt.savefig(
        os.path.join(out_path, f'adapt--{data}{out_suffix}.png'),
        bbox_inches='tight'
    )

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()

