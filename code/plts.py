""""Plotting functions for the oscillation methods project."""

import os

import matplotlib.pyplot as plt
import seaborn as sns

from utils import AVG_FUNCS_NAN
from settings import PLT_EXT

###################################################################################################
###################################################################################################

def plot_bar(d1, d2, label1=None, label2=None, err=None, width=0.65,
             average='mean', lw=4, figsize=None, **plt_kwargs):
    """Plot a bar graph."""

    _, ax = plt.subplots(figsize=figsize)

    if err:
        err = [np.nanstd(d1), np.nanstd(d2)]

    avg_func = AVG_FUNCS_NAN[average]

    ax.bar([0.5, 1.5], [avg_func(d1), avg_func(d2)], yerr=err,
            tick_label=[label1, label2], width=width, **plt_kwargs)

    ax.set_xlim([0, 2])

    ax.set_yticks([]);
    if not label1: ax.set_xticks([]);

    sns.despine(ax=ax, top=True, right=True)
    [ax.spines[side].set_linewidth(lw) for side in ['left', 'bottom']]


def plot_estimates(xdata, ydata, inds=None, colors=None,
                   xlim=None, figsize=None):
    """Plot estimates across a range of signal values."""

    _, ax = plt.subplots(figsize=figsize)
    ax.plot(xdata, ydata, color='black')
    ax.plot(xdata, ydata, '.', ms=25, color='black', markeredgecolor='w')

    if inds:
        for ind, color in zip(range(len(inds)), list(reversed(colors))):
            ax.plot([xdata[inds[ind]]], [ydata[inds[ind]]],
                    '.', color=color, ms=30, markeredgecolor='w')
    _clear_x(ax)
    if xlim:
        ax.set_xlim(xlim)


def savefig(save_fig, name, folder='figures/', ext=PLT_EXT, **kwargs):
    """Helper function for saving plots."""

    if save_fig:
        plt.savefig(os.path.join(folder, name + '.' + ext), bbox_inches='tight', **kwargs)


def style_psd(ax, clear_x=False, clear_y=True, line_colors=None, line_alpha=1, fontsize=25):
    """Aesthetic styling for a power spectrum plot."""

    ax.set_xlabel(''); ax.set_ylabel('');

    if line_colors:
        for line, color in zip(ax.get_lines(), line_colors):
            line.set_color(color)
            line.set_alpha(line_alpha)

    if clear_x:
        _clear_x(ax)
    else:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)

    if clear_y:
        _clear_y(ax)
    else:
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)


def _clear_x(ax):

    for xlabel_i in ax.get_xticklabels():
        xlabel_i.set_visible(False)
        xlabel_i.set_fontsize(0.0)
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1line.set_visible(False)

def _clear_y(ax):

    for ylabel_i in ax.get_yticklabels():
        ylabel_i.set_visible(False)
        ylabel_i.set_fontsize(0.0)
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1line.set_visible(False)
