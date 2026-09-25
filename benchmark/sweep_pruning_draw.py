"""Plot success-rate heatmaps from the LIBERO pruning sweep CSV."""

import argparse
from fractions import Fraction
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=Path('results/libero/pruning_sweep/success_rates.csv'))
    parser.add_argument('--output', type=Path, help='Defaults to heatmap.png beside the CSV')
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    groups = list(df.groupby(['suite', 'task_id'], sort=False))
    if not groups:
        parser.error('CSV contains no tasks')

    rates = sorted({Fraction(str(value)) for value in df['keep_fraction']})
    steps = sorted(df['num_steps'].unique())
    df['keep_rate'] = df['keep_fraction'].map(lambda value: Fraction(str(value)))
    # Incomplete evaluations must not appear as zero-success evaluations.
    df.loc[df['status'] != 'complete', 'success_rate'] = np.nan
    groups = list(df.groupby(['suite', 'task_id'], sort=False))
    ncols = min(2, len(groups))
    fig, axes = plt.subplots((len(groups) + ncols - 1) // ncols, ncols,
                             figsize=(6 * ncols, 5 * ((len(groups) + ncols - 1) // ncols)),
                             squeeze=False, layout='constrained')
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#dddddd')
    for ax, ((suite, task_id), task_df) in zip(axes.flat, groups):
        values = task_df.pivot(index='keep_rate', columns='num_steps', values='success_rate')
        values = values.reindex(index=rates, columns=steps).to_numpy(dtype=float)
        plot = ax.imshow(np.ma.masked_invalid(values), origin='lower', aspect='auto',
                         cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f'{suite} / task {task_id}\n' + textwrap.fill(task_df.iloc[0]['task'], 48), fontsize=10)
        ax.set_xticks(range(len(steps)), labels=[str(step) for step in steps])
        ax.set_yticks(range(len(rates)), labels=[str((rate) * Fraction(3, 2)) for rate in rates])
        ax.set_xlabel('Decoding steps')
        ax.set_ylabel('Fraction of Visual Tokens')
        for row, column in np.ndindex(values.shape):
            value = values[row, column]
            ax.text(column, row, '—' if np.isnan(value) else f'{value:.0%}',
                    ha='center', va='center',
                    color='white' if np.isfinite(value) and value < 0.5 else 'black')
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    fig.colorbar(plot, ax=list(axes.flat)[:len(groups)], label='Success rate',
                 format=PercentFormatter(xmax=1), shrink=0.8)
    fig.suptitle('LIBERO pruning sweep\nGrey / —: incomplete or unavailable', fontsize=13)
    output = args.output or args.csv.with_name('heatmap.png')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
