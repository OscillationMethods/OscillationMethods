import numpy as np
import matplotlib.pyplot as plt

from fooof.utils import trim_spectrum
from neurodsp.filt import filter_signal
from neurodsp.sim import sim_oscillation, sim_combined, sim_powerlaw
from neurodsp.spectral import compute_spectrum
from neurodsp.utils import create_times

import bycycle.features
import seaborn as sns

np.random.seed(3)
n_seconds = 25
fs = 1000

times = create_times(n_seconds, fs)
exp = -2.0
ap_filt = (2, 150)

cf = 10
rdsym = 0.5
psd_range = (3, 35)
alpha_range = (8, 12)
beta_range = (15, 35)
rdsyms = [0.50, 0.625, 0.75, 0.875, 1.0]
comps = {
    "sim_powerlaw": {"exponent": exp, "f_range": ap_filt},
    "sim_oscillation": {"freq": cf, "cycle": "asine", "rdsym": rdsym},
}

# Define relative power of the signal components
comp_vars = [1, 0.8]
plt_kwargs = {"xlabel": "", "ylabel": "", "alpha": [0.75, 0.75]}

# Simulate an asymmetric oscillation
fig, ax = plt.subplots(4, 1, figsize=(4, 4))
rdsyms = np.arange(0.5, 0.8, 0.15)
noise = sim_powerlaw(n_seconds, fs, exponent=-2.5)

all_pows = []
rdsyms = [0.50, 0.625, 0.75, 0.875]
labels = ["rd-sym=%.3f" % i for i in rdsyms]

burst_detection_kwargs = {
    "amplitude_fraction_threshold": 0.0,
    "amplitude_consistency_threshold": 0.0,
    "period_consistency_threshold": 0.0,
    "monotonicity_threshold": 0.0,
}

cmap = [plt.cm.viridis(i) for i in np.linspace(0, 1, len(rdsyms) + 1)]
cmap = cmap[::-1]

for i, rdsym in enumerate(rdsyms):
    osc = sim_oscillation(n_seconds, fs, cf, cycle="asine", rdsym=rdsym)

    # compute peak deviation
    df = bycycle.features.compute_features(
        osc,
        fs,
        f_range=(cf - 2, cf + 2),
        burst_detection_kwargs=burst_detection_kwargs,
    )

    sig_filt_al = filter_signal(osc, fs, "bandpass", alpha_range)
    df_filt = bycycle.features.compute_features(
        sig_filt_al,
        fs,
        f_range=(cf - 2, cf + 2),
        burst_detection_kwargs=burst_detection_kwargs,
    )

    ax_time = ax[i]
    ax_time.plot(times, sig_filt_al, color="k", lw=2)
    ax_time.plot(times, osc, color=cmap[i], lw=2)

    ax_time.plot(
        times[df.sample_peak], df.volt_peak, ".", color=cmap[i], markersize=10
    )
    ax_time.plot(
        times[df_filt.sample_peak], df_filt.volt_peak, "r.", markersize=10
    )

    # cheating because the noise is added only here
    osc += 20 * noise
    cur_freqs, cur_pows = compute_spectrum(osc, fs, nperseg=1.5 * fs)
    all_pows.append(cur_pows)
    ax_time.axis("off")
    ax_time.set(xlim=(4.5, 5), ylim=(-1.6, 1.6))

plt.savefig("../figures/05-ts_all.pdf", bbox_inches="tight", dpi=300)


# plot spectra
fig, ax = plt.subplots(2, 1, figsize=(3, 4))
freqs, psd = trim_spectrum(cur_freqs, np.array(all_pows), psd_range)

ax_psd = ax[0]
for i in range(4):
    ax_psd.semilogy(freqs, psd[i].T, color=cmap[i], lw=2)
ax_psd.set(xlabel="frequency [Hz]", xlim=(5, 35), yticks=[], ylabel="log PSD")
sns.despine(ax=ax_psd)

# simulate for a number of trials
nr_trials = 50
rdsyms = np.arange(0.5, 1, 0.025)
beta_pows = np.zeros((len(rdsyms), nr_trials))

for i_rdsym, rdsym in enumerate(rdsyms):

    for i_trial in range(nr_trials):

        # Create the signal
        comps = {
            "sim_powerlaw": {"exponent": exp, "f_range": ap_filt},
            "sim_oscillation": {"freq": cf, "cycle": "asine", "rdsym": rdsym},
        }

        cur_sig = sim_combined(n_seconds, fs, comps, comp_vars)

        # Compute the spectrum and collect the measured beta power
        cur_freqs, cur_pows = compute_spectrum(cur_sig, fs, nperseg=fs)
        _, beta_pow = trim_spectrum(cur_freqs, cur_pows, beta_range)
        beta_pow = np.mean(beta_pow)
        beta_pows[i_rdsym, i_trial] = beta_pow


mean_beta = np.mean(beta_pows, axis=1)
std_beta = np.std(beta_pows, axis=1)

ax_sym = ax[1]
ax_sym.plot(rdsyms, mean_beta, "k.-")

rdsyms_selected = [0.50, 0.625, 0.75, 0.875]
for i, rd in enumerate(rdsyms_selected):
    idx = np.argmin(np.abs(rdsyms - rd))
    ax_sym.plot(
        rdsyms[idx],
        mean_beta[idx],
        ".",
        color=cmap[i],
        markersize=10,
        markeredgecolor="w",
    )


# solely for prettier error spans
rdsyms = np.hstack([0.48, rdsyms, 1.0])
mean_beta = np.hstack([mean_beta[0], mean_beta, mean_beta[-1]])
std_beta = np.hstack([std_beta[0], std_beta, std_beta[-1]])

ax_sym.fill_between(
    rdsyms,
    mean_beta - std_beta,
    mean_beta + std_beta,
    facecolor="k",
    alpha=0.25,
    edgecolor=None,
)
ax_sym.set(
    xlabel="rise-decay asymmetry",
    xticks=np.arange(0.5, 1.01, 0.1),
    ylabel="average beta power",
    yticks=[],
    xlim=(0.48, 1.0),
)
sns.despine(ax=ax_sym)
fig.tight_layout()
fig.savefig("../figures/05-psd_shapes.pdf", bbox_inches="tight")
