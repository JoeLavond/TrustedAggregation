""" Packages """
# base
import argparse
import numpy as np
import os
from pathlib import Path

# visual
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='cifar', type=str)
    parser.add_argument('--n_classes', default=100, type=int)

    parser.add_argument('--alpha', default=1, type=int)
    parser.add_argument('--alpha_val', default=10000, type=int)

    parser.add_argument('--n_rounds', default=250, type=int)

    parser.add_argument('--neuro', default=1, type=int)
    parser.add_argument('--neuro_p', default=0.1, type=float)

    parser.add_argument('--font_size', default=14, type=int)
    parser.add_argument('--show', default=1, type=int)

    return parser.parse_args()


""" Main function HERE """


def main():
    args = get_args()
    args.d_rounds = args.n_rounds

    out_suffix = (
        (f'--neuro_p{args.neuro_p}' if args.neuro else '')
        + (f'--alpha{args.alpha}' if args.alpha != 10000 else '')
    )

    font = {
        'size': args.font_size
    }
    mpl.rc('font', **font)

    line_style = 'solid'
    cool_color = '#88CCEE'
    warm_color = '#EE7733'

    out_path = './extra'
    if not os.path.exists('./extra'):
        os.makedirs('./extra')

    d_scales = [2, 1.5, 1.25, 1.1]
    n_maliciouses = [1, 2, 4]
    fig, axarr = plt.subplots(
        nrows=len(d_scales), ncols=len(n_maliciouses), figsize=(len(n_maliciouses) * 4, len(d_scales) * 4), sharey=True, sharex=True
    )

    for i, d_scale in enumerate(d_scales):
        file_suffix = f'--d_scale{d_scale}' if d_scale != 2. else ''

        for j, n_malicious in enumerate(n_maliciouses):
            subdir = f'--d_start1--m_start1--n_malicious{n_malicious}'
            axarr[i, j].set_ylim(-0.05, 1.05)

            if i == 0:
                axarr[0, j].set_title(
                    f'{n_malicious * 10}% Malicious Users'
                    + (', NT' if args.neuro else '')
                )

            if j == 0:
                axarr[i, 0].set_ylabel(f'theta = {d_scale}')

            if i == len(d_scales) - 1:
                axarr[i, j].set_xlabel('Communication Round')

            path = os.path.join(
                f'{Path.home()}',
                'Documents',
                'TAG',
                f'{args.data}_{args.n_classes}',
                ('neuro' if args.neuro else 'classic'),
                'tag',
                ('distributed' if (n_malicious > 1) else 'centralized'),
                'alpha' + str(args.alpha) + '--alpha_val' + str(args.alpha_val)
            )

            temp_global = np.load(
                os.path.join(
                    path,
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
                ), allow_pickle=True
            )

            axarr[i, j].plot(range(0, args.d_rounds + 1), temp_global[:args.d_rounds + 1, 1], c=cool_color,
                     linestyle=line_style, label='Classification Accuracy')
            axarr[i, j].plot(range(0, args.d_rounds + 1), temp_global[:args.d_rounds + 1, 2], c=warm_color,
                     linestyle=line_style, label='Attack Success Rate')

            if i == 0 and j == 0:
                axarr[i, j].legend()

    handles, labels = axarr[0, 0].get_legend_handles_labels()
    axarr[0, 0].get_legend().remove()
    fig.legend(handles, labels, ncol=2, bbox_to_anchor=(0.5, 0.03), loc='lower center')

    plt.savefig(
        os.path.join(out_path, f'sensitivity{out_suffix}.png'),
        bbox_inches='tight'
    )
    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
