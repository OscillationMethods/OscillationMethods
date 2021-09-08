# Oscillation Methods

`Oscillation Methods` project repository: methodological considerations for studying neural oscillations.

[![Paper](https://img.shields.io/badge/DOI-10.1111/ejn.15361-informational.svg)](https://doi.org/10.1111/ejn.15361)

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

This project is described in the following paper:

    Donoghue T, Schaworonkow N, & Voytek B (2021). Methodological considerations for
    studying neural oscillations. European Journal of Neuroscience. DOI: 10.1111/ejn.15361

Direct Link: https://onlinelibrary.wiley.com/doi/10.1111/ejn.15361

## Requirements

If you want to re-run this project, you can install the required dependencies and re-run the notebooks.

This repository requires Python (>=3.6), and standard scientific packages.

This project also requires the following additional packages:

- [neurodsp](https://github.com/neurodsp-tools/neurodsp) >= 2.2.0
- [bycycle](https://github.com/bycycle-tools/bycycle) >= 0.1.3
- [fooof](https://github.com/fooof-tools/fooof) >= 1.0.0

The general set of requirements is listed in `requirements.txt`.
Note that some notebooks have additional requirements, that are listed in the notebook.
