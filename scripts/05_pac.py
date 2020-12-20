# small demo for how waveform shape leads to spurious PAC

import numpy as np
import matplotlib.pyplot as plt
import neurodsp.spectral as spec
import neurodsp.filt.filter as filter1
import scipy.signal
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def mu_wave(time, shift=0, main_freq=10):
    """
    Create a non-sinusoidal signal as a sum of two sine-waves with fixed
    phase-lag.

    Parameters:
    ----------
    time : array, time interval in seconds.
    shift : sets initial phase of oscillation
    wave_shift : float, phase lag in radians of faster oscillation to slower.
    main_freq : float, base frequency of oscillation.

    Returns:
    --------
    alpha : array, signal over time, base frequency
    beta : array, signal over time, harmonic frequency

    """
    amp_A = 1.0
    amp_B = 0.25
    shift = 0
    wave_shift = 0.5 * np.pi
    alpha = amp_A * np.sin(main_freq * 2 * np.pi * (time + shift))
    beta = amp_B * np.sin(
        main_freq * 2 * np.pi * 2 * (time + shift) + wave_shift
    )

    return alpha, beta


# change colors to fit the rest of figure
mu_color = "k"
sinus_color = "r"
harmonic_color = "b"

# create signals
fs = 1000
base_frequency = 10
time = np.arange(0, 2, 0.001)
signal_alpha, signal_beta = mu_wave(time, 0, base_frequency)
signal_mu = signal_alpha + signal_beta

signal_mu_filt = filter1.filter_signal(
    signal_mu, fs, "bandpass", [15, 25], remove_edges=False
)

# regulat filter order for changing attenuation for flanks
signal_beta_filt = filter1.filter_signal(
    signal_beta, fs, "bandpass", [17.5, 22.5], n_cycles=3, remove_edges=False
)
signal_alpha_filt = filter1.filter_signal(
    signal_alpha, fs, "bandpass", [15, 25], remove_edges=False
)


# spectrum
fmax = 50
freqs, spec_mu = spec.compute_spectrum(signal_mu, fs, f_range=(2, fmax))
freqs, spec_beta = spec.compute_spectrum(signal_beta, fs, f_range=(2, fmax))
freqs, spec_beta_filt = spec.compute_spectrum(
    signal_beta_filt, fs, f_range=(2, fmax)
)
freqs, spectrum_alpha = spec.compute_spectrum(
    signal_alpha, fs, f_range=(2, fmax)
)

beta_env = np.abs(scipy.signal.hilbert(signal_beta_filt))
mu_env = np.abs(scipy.signal.hilbert(signal_mu_filt))
phase_alpha = np.angle(scipy.signal.hilbert(signal_alpha))

n_bins = 21
bins = np.linspace(-np.pi, np.pi, n_bins)
phase_bins = np.digitize(phase_alpha, bins)

pac = np.zeros((n_bins, 2))
for i_bin, bin in enumerate(np.unique(phase_bins)):
    pac[i_bin, 0] = np.mean(mu_env[(phase_bins == bin)])
    pac[i_bin, 1] = np.mean(beta_env[(phase_bins == bin)])

# cosmetics
offset = 0.27
tmax = 0.6

# plot
fig, ax = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.8, 1]})
ax_ts = ax[0]
ax_ts.axhline(0, color="k", alpha=0.2)
ax_ts.plot(
    time - offset, signal_mu, color=mu_color, label="non-sinusoidal waveform"
)
ax_ts.plot(
    time - offset,
    signal_mu_filt,
    color=harmonic_color,
    label="bandpass-filtered in beta-band",
)
ax_ts.plot(
    time-offset, signal_beta, color=sinus_color, label="sinusoid in beta-band"
)

ax_ts.set_title("time-domain signal")
ax_ts.set_xlim(0, tmax-offset)
ax_ts.set_xlabel("time [s]")
ax_ts.set_ylim(-2.3, 1)
ax_ts.set_yticks([-1, 0, 1])
ax_ts.legend(loc="lower right", fontsize=9)

# tiny spectrum
axins = inset_axes(ax_ts, width="25%", height="25%", loc=3, borderpad=1.25)
axins.semilogy(freqs, spec_mu, color=mu_color, alpha=0.7)
axins.semilogy(freqs, spec_beta, color=sinus_color, alpha=0.9)
axins.semilogy(freqs, spec_beta_filt, color=harmonic_color, alpha=0.9)
axins.set_xlim(0, fmax)
axins.set_yticks([])
axins.set_xticks([])

# PAC
ax_pac = ax[1]
ax_pac.plot(bins, pac[:, 1], color=sinus_color)
ax_pac.plot(bins, pac[:, 0], color=harmonic_color)
ax_pac.set_title("phase-amplitude coupling")
ax_pac.set_xticks(np.linspace(-np.pi, np.pi, 5))
ax_pac.set_xticklabels([r"-$\pi$", r"-$0.5\pi$", "0", r"$0.5\pi$", r"$\pi$"])
ax_pac.set_xlabel("alpha phase")
ax_pac.set_ylabel("mean beta envelope")
ax_pac.set_xlim(-np.pi, np.pi)

fig.set_size_inches(8, 3.5)
fig.tight_layout()
fig.savefig("../figures/demo_pac_waveform.pdf", dpi=300)
fig.show()
