""""Plotting functions for the oscillation methods project."""

import os

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np

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


def plot_band_changes(deltas, colors=None, ylim=None):
    """Plot differences in power across bands."""

    labels = [r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$', r'$\gamma$']

    plt.bar([0, 1, 2, 3, 4], deltas.values(),
            tick_label=labels, color=colors, alpha=0.6)
    if ylim:
        plt.ylim(ylim);


def plot_spectrogram(times, freqs, pxx, flim=None, clear_ticks=False, figsize=None):
    """Helper function for plotting spectrograms."""

    _, ax = plt.subplots(figsize=figsize)
    plt.imshow(pxx[0:flim, :], extent=[times[0], times[-1], freqs[0], freqs[flim]],
           aspect='auto', origin='lower', interpolation='hanning')

    if clear_ticks:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


def plot_waveform(times, osc, filt, df_raw, df_filt,
                  color=None, xlim=None, ylim=None, ax=None):
    """Plot a time series, with cycle points annotated."""

    ax.plot(times, filt, color="k", lw=2)
    ax.plot(times, osc, color=color, lw=2)

    ax.plot(times[df_raw.sample_peak], df_raw.volt_peak, ".", color=color, markersize=10)
    ax.plot(times[df_filt.sample_peak], df_filt.volt_peak, "r.", markersize=10)

    ax.axis("off")
    ax.set(xlim=xlim, ylim=ylim)


def plot_power_by_shape(rdsyms, mean_beta, std_beta, selected, cmap):
    """Plot measured oscillatory power across waveform shape values."""""

    _, ax_sym = plt.subplots(figsize=(5, 3))
    ax_sym.plot(rdsyms, mean_beta, "k.-")

    for ind, rd in enumerate(selected):
        idx = np.argmin(np.abs(rdsyms - rd))
        ax_sym.plot(rdsyms[idx], mean_beta[idx], ".",
                    color=cmap[ind], markersize=15, markeredgecolor="w")

    rdsyms = np.hstack([0.48, rdsyms, 1.0])
    mean_beta = np.hstack([mean_beta[0], mean_beta, mean_beta[-1]])
    std_beta = np.hstack([std_beta[0], std_beta, std_beta[-1]])

    ax_sym.fill_between(rdsyms, mean_beta - std_beta, mean_beta + std_beta,
                        facecolor="k", alpha=0.25, edgecolor=None)
    ax_sym.set(xlabel="rise-decay asymmetry", xticks=np.arange(0.5, 1.01, 0.1),
               ylabel="average beta power", yticks=[], xlim=(0.48, 1.0))
    sns.despine(ax=ax_sym)


def plot_harmonic_power(time, signal_mu, signal_mu_filt, signal_beta, colors):
    """Plot a signal with filtered harmonic bands."""

    offset = 0.275
    tmax = 0.58

    _, ax_ts = plt.subplots(figsize=(7, 4))

    ax_ts.axhline(0, color="k", alpha=0.2)
    ax_ts.plot(time - offset, signal_mu, color=colors[0])
    ax_ts.plot(time - offset, signal_mu_filt, color=colors[1])
    ax_ts.plot(time-offset, signal_beta, color=colors[2])

    ax_ts.set(xlim=(0, tmax-offset), ylim=(-1.4, 1), xlabel="time [s]", yticks=[-1, 0, 1]);


def plot_pac(bins, pac, colors):
    """Plot phase-amplitude coupling."""

    _, ax_pac = plt.subplots(figsize=[4, 4])

    ax_pac.plot(bins, pac[:, 1], color=colors[0])
    ax_pac.plot(bins, pac[:, 0], color=colors[1])

    ax_pac.set_xticks(np.linspace(-np.pi, np.pi, 5))
    ax_pac.set_xticklabels([r"-$\pi$", r"-$0.5\pi$", "0", r"$0.5\pi$", r"$\pi$"])
    ax_pac.set_xlabel("alpha phase")
    ax_pac.set_ylabel("mean beta envelope")
    ax_pac.set_xlim(-np.pi, np.pi);


def plot_data_grid(times, data, freq, psds, colors, freq1=None, freq2=None):
    """Plot a grid of data, showing power spectra and time series."""

    fig = plt.figure(figsize=(7.75, 4.5))
    ax_grid = GridSpec(3, 2, figure=fig, width_ratios=[1, 1.2],
                       top=0.95, bottom=0.1, left=0.0, right=0.95, wspace=0.25)

    for ind, (ts, psd, color) in enumerate(zip(data, psds, colors)):

        ax_psd = fig.add_subplot(ax_grid[ind, 0])
        ax_psd.set(xlim=(freq[0], freq[-1]), xticks=[0, 10, 20, 30])
        ax_psd.semilogy(freq, psd, color=color)
        ax_psd.set_yticks([])

        if freq1 and freq2:
            if ind != 1:
                ax_psd.axvline(freq1, color=colors[0], alpha=0.5)
                ax_psd.axvline(2 * freq1, color=colors[0], alpha=0.5)

            if ind > 0:
                ax_psd.axvline(freq2, color=colors[1], alpha=0.5)
                ax_psd.axvline(2 * freq2, color=colors[1], alpha=0.5)

        ax_ts = fig.add_subplot(ax_grid[ind, 1])
        ax_ts.plot(times, ts, color=color)
        ax_ts.set(xlim=(0, 1), yticks=[], ylim=(ts.min() - 0.1, ts.max() + 0.1))

        if ind < 2:
            ax_psd.set_xlabel(None)
            ax_psd.set_xticks([])
            ax_ts.set_xlabel(None)
            ax_ts.set_xticks([])

        sns.despine(ax=ax_psd)
        sns.despine(ax=ax_ts, left=True)


def plot_legend(labels, colors):
    """Helper function to create a figure legend, by itself."""

    ff = lambda mark, col: plt.plot([], [], marker=mark, color=col, ls="none")[0]
    handles = [ff("s", colors[ind]) for ind in range(len(labels))]

    legend = plt.legend(handles, labels, loc=10, framealpha=1, frameon=False)
    plt.axis('off')


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
