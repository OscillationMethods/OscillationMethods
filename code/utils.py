"""Analysis utilities for the oscillation methods project."""

import numpy as np
import scipy.signal

from fooof.utils import trim_spectrum

###################################################################################################
###################################################################################################

AVG_FUNCS = {
    'mean' : np.mean,
    'median' : np.median,
    'sum' : np.sum
}


AVG_FUNCS_NAN = {
    'mean' : np.nanmean,
    'median' : np.nanmedian,
}

def compute_abs_power(freqs, powers, band, method='sum'):
    """Compute absolute power for a given frequency band."""

    _, band_powers = trim_spectrum(freqs, powers, band)
    avg_power = AVG_FUNCS[method](band_powers)

    return avg_power


def compute_rel_power(freqs, powers, band, method='sum', norm_range=None):
    """Compute relative power for a given frequency band."""

    band_power = compute_abs_power(freqs, powers, band, method)

    total_band = [freqs.min(), freqs.max()] if not norm_range else norm_range
    total_power = compute_abs_power(freqs, powers, total_band, method)

    rel_power = band_power / total_power * 100

    return rel_power


def phase_locking_value(theta1, theta2):
    """Compute the phase locking value between two signals.
    From: https://dsp.stackexchange.com/questions/25165/phase-locking-value-phase-synchronization
    """

    complex_phase_diff = np.exp(np.complex(0, 1) * (theta1 - theta2))
    plv = np.abs(np.sum(complex_phase_diff)) / len(theta1)

    return plv


def get_components(cf, exp, ap_filt):
    """Helper function for defining combined signals."""

    return {'sim_powerlaw' : {'exponent' : exp, 'f_range' : ap_filt},
            'sim_oscillation' : {'freq' : cf}}


def rotate_sig(sig, fs, delta_exp, f_rotation):
    """Spectrally rotate a time series."""

    fft_vals = np.fft.fft(sig)
    f_axis = np.fft.fftfreq(len(sig), 1./fs)

    if f_axis[0] == 0:
        skipped_zero = True
        p_0 = fft_vals[0]
        f_axis, fft_vals = f_axis[1:], fft_vals[1:]

    else:
        skipped_zero = False

    f_mask = 10**(np.log10(np.abs(f_axis)) * delta_exp)
    f_mask = f_mask / f_mask[np.where(f_axis == f_rotation)]

    fft_rot = fft_vals * f_mask

    if skipped_zero:
        fft_rot = np.insert(fft_rot, 0, p_0)

    sig_out = np.real(np.fft.ifft(fft_rot))

    return sig_out


def make_osc_def(n_off1, n_on, n_off2):
    """Create an oscillation definition of off/on/off."""

    return np.array([False] * n_off1 + [True] * n_on + [False] * n_off2)


def mu_wave(time, shift=0, main_freq=10, wave_shift=0.5*np.pi,
            amp_alpha=1.0, amp_beta=0.25, comb=True):
    """Create a non-sinusoidal signal as a sum of two sine-waves with fixed phase-lag.

    Parameters:
    ----------
    time : array, time interval in seconds.
    shift : sets initial phase of oscillation
    wave_shift : float, phase lag in radians of faster oscillation to slower.
    main_freq : float, base frequency of oscillation.

    Returns:
    --------
    signal : array, non-sinusoidal signal over time
    """

    alpha = amp_alpha * np.sin(main_freq * 2 * np.pi * (time + shift))
    beta = amp_beta * np.sin(main_freq * 2 * np.pi * 2 * (time + shift) + wave_shift)

    if not comb:
        return alpha, beta
    else:
        return alpha + beta


def compute_pac(signal_mu_filt, signal_beta_filt, signal_alpha, n_bins=21):
    """Compute phase-amplitude coupling for a mu signal."""

    beta_env = np.abs(scipy.signal.hilbert(signal_beta_filt))
    mu_env = np.abs(scipy.signal.hilbert(signal_mu_filt))
    phase_alpha = np.angle(scipy.signal.hilbert(signal_alpha))

    bins = np.linspace(-np.pi, np.pi, n_bins)
    phase_bins = np.digitize(phase_alpha, bins)

    pac = np.zeros((n_bins, 2))
    for i_bin, c_bin in enumerate(np.unique(phase_bins)):
        pac[i_bin, 0] = np.mean(mu_env[(phase_bins == c_bin)])
        pac[i_bin, 1] = np.mean(beta_env[(phase_bins == c_bin)])

    return bins, pac
