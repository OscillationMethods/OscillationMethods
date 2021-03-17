# Oscillation Methods

`Oscillation Methods` project repository: methodological considerations for studying neural oscillations.

## Overview

This project is a overview of methodological considerations for analyzing neural oscillations.

Using simulated data, we explore the relationship between data properties and common analysis approaches, highlighting potential issues, organized into a collection of 7 methodological considerations.

These methodological considerations are:
- #1) verifying the presence of oscillations
- #2) band definitions
- #3) aperiodic activity
- #4) temporal variability
- #5) waveform shape
- #6) overlapping rhythms / source separation
- #7) power confounds / signal-to-noise ratio

Each topic is covered by a notebook in this repository.

## Reference

This project is described in the following preprint:

    Donoghue T, Schaworonkow N, & Voytek B. Methodological considerations for
    studying neural oscillations. PsyArXiv (LINK FORTHCOMING).

A poster of this project was presented at the
[LiveMEEG](https://livemeeg2020.org/)
and is available
[here](https://www.dropbox.com/s/jz9fpdk4v8am18h/Donoghue%26Voytek-ConsiderationsMeasuringNeuralOscillations.pdf?dl=0).

## Requirements

If you want to re-run this project, you can install the required dependencies and re-run the notebooks.

This repository requires Python (>=3.6), and standard scientific packages.

This project also requires the following additional packages:

- [neurodsp](https://github.com/neurodsp-tools/neurodsp) >= 2.1.0
- [bycycle](https://github.com/bycycle-tools/bycycle) >= 1.0.0
- [fooof](https://github.com/fooof-tools/fooof) >= 1.0.0

The full set of requirements are listed in `requirements.txt`.
