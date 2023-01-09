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

    parser.add_argument('--neuro', default=0, type=int)
    parser.add_argument('--neuro_p', default=0.1, type=float)

    parser.add_argument('--beta', default=0.2, type=float)
    parser.add_argument('--d_scale', default=2, type=float)
    parser.add_argument('--d_smooth', default=1, type=float)

    parser.add_argument('--n_rounds', default=250, type=int)
    parser.add_argument('--d_rounds', default=None, type=int)

    parser.add_argument('--font_size', default=12, type=int)
    parser.add_argument('--show', default=1, type=int)

    return parser.parse_args()


""" Main function HERE """
def main():

    args = get_args()
    out_suffix = (
        f'--beta{args.beta}'
        + (f'--d_scale{args.d_scale}' if args.d_scale != 2. else '')
        + (f'--neuro_p{args.neuro_p}' if args.neuro else '')
        + (f'--n_val_data{args.n_val_data}' if args.n_val_data is not None else '')
        + (f'--alpha{args.alpha}' if args.alpha != 10000 else '')
        + (f'--alpha_val{args.alpha_val}' if args.alpha_val != 10000 else '')
        + ('--no_smooth' if not args.d_smooth else '')
        + ('--vgg' if not args.resnet else '')
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

    methods = ('base/mean', 'base/median', 'base/trust')
    file_suffices = (
        f'--beta{args.beta}',
        '',
        ''
    )

    line_styles = ['dotted', 'dashed', 'dashdot']
    cool_colors = sns.color_palette("bright").as_hex()
    cool_colors = cool_colors[1:(len(methods) + 1)]
    warm_colors = sns.color_palette("dark").as_hex()
    warm_colors = warm_colors[1:(len(methods) + 1)]

    out_path = './extra'
    if not os.path.exists('./extra'):
        os.makedirs('./extra')


    """ Global Accuracy """
    fig, axarr = plt.subplots(ncols=3, figsize=(3*4, 4))

    dbas = [0, 1, 1]
    n_maliciouses = [1, 2, 4]
    titles = [
        '10% Malicious', 'DBA 20% Malicious', 'DBA 40% Malicious'
    ]

    for i, (dba, n_malicious) in enumerate(zip(dbas, n_maliciouses)):
        plt.sca(axarr[i])

        subdirs = [
            f'--m_start{args.m_start}--n_malicious{n_malicious}',
            f'--m_start{args.m_start}--n_malicious{n_malicious}',
            f'--m_start{args.m_start}--n_malicious{n_malicious}'
        ]


        plt.title(titles[i], fontsize=1.5*args.font_size)
        plt.xlabel('Communication Round')

        clean_lines = []
        pois_lines = []

        for j, method in enumerate(methods):

            path = os.path.join(
                f'{Path.home()}/fed-tag',
                data,
                ('neuro' if args.neuro else 'classic'),
                method,
                ('distributed' if dba else 'centralized'),
                'alpha' + str(args.alpha) + '--alpha_val' + str(args.alpha_val)
            )
            alt_path = os.path.join(
                f'{Path.home()}/fed-tag',
                data,
                ('neuro' if args.neuro else 'classic'),
                method,
                ('distributed' if dba else 'centralized'),
                'alpha' + str(args.alpha) + '--alpha_val10000'
            )
            f_path = os.path.join(
                (
                    f'n_rounds{args.n_rounds}'
                    + subdirs[j]
                ),
                'data',
                (
                    'output_global_acc'
                    + (f'--neuro_p{args.neuro_p}' if args.neuro else '')
                    + file_suffices[j]
                    + ('--vgg' if not args.resnet else '')
                    + '.npy'
                )
            )
            print(f_path)

            try:
                temp_global = np.load(
                    os.path.join(
                        path, f_path
                    ), allow_pickle=True
                )
            except:
                temp_global = np.load(
                    os.path.join(
                        alt_path, f_path
                    ), allow_pickle=True
                )

            clean_line, = plt.plot(range(0, args.d_rounds + 1), temp_global[:args.d_rounds + 1, 1], c=cool_colors[j], linestyle=line_styles[j], label=method)
            clean_lines.append(clean_line)
            pois_line, = plt.plot(range(0, args.d_rounds + 1), temp_global[:args.d_rounds + 1, 2], c=warm_colors[j], linestyle=line_styles[j], label=method)
            pois_lines.append(pois_line)

    # create legend
    out_methods = ['Coordinate Median', 'Coordinate Trim-Mean', 'FLTrust']
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
        os.path.join(out_path, f'baseline--{data}{out_suffix}.png'),
        bbox_inches='tight'
    )

    if args.show:
        #plt.tight_layout()
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()

