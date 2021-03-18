"""Analysis utilities for the oscillation methods project."""

import numpy as np

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


def mu_wave(time, shift=0, wave_shift=1, main_freq=10):
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

    amp_A = 1.0
    amp_B = 0.25

    alpha = amp_A * np.sin(main_freq * 2 * np.pi * (time + shift))
    beta = amp_B * np.sin(
        main_freq * 2 * np.pi * 2 * (time + shift) + wave_shift
    )

    signal = alpha + beta

    return signal
