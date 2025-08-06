# -*- coding: utf-8 -*-
"""
despiking.py: Raman spectrum despiking functions
@author: Ondrej Vaculik
@email: vacuon@isibrno.cz
"""

import numpy as np
import pandas as pd
from nptyping import NDArray
from scipy.signal import convolve

# functions


def find_peak(spectrum: NDArray, threshold: float):
    """Detect peak using convolution with Dirichlet kernel.
    https://www.sciencedirect.com/science/article/pii/S0169743916300600

    Args:
        spectrum (np.ndarray): Raman spectrum array
        threshold (float): Detection threshold

    Returns:
        int: Strongest peak position located in spectrum.
    """
    d2x_kernel = [-1, 2, -1]
    normalized = spectrum / np.max(spectrum)
    convolution = convolve(normalized, d2x_kernel)
    conv_fixed = np.pad(convolution[3:-3], (3, 3), mode="edge")
    if np.max(conv_fixed) > threshold:
        return np.argmax(conv_fixed) - 1  # Shift to peak maximum


def remove_peak(spectrum: NDArray, iterations=1, window=8, threshold=0.2):
    """Remove cosmic peaks from Raman spectrum

    Args:
        spectrum (np.ndarray): Raman spectrum array
        iterations (int, optional): Number of repeated iterations. Defaults to 1.
        window (int, optional): Width of area that will be replaced. Defaults to 3.
        threshold (float, optional): Peak sensitivity. Defaults to 0.25.

    Raises:
        ValueError: Iterations must be positive integer

    Returns:
        np.ndarray: Despiked Raman spectrum
    """
    corrected = spectrum.copy().astype(np.float_)
    peak_index = find_peak(spectrum, threshold)
    if peak_index:
        corrected[int(peak_index - window // 2): int(peak_index + window // 2)] = np.nan
        corrected = (
            pd.Series(corrected).interpolate(limit_direction="both").to_numpy()
        )  # to avoid edge cases
    if iterations == 1:
        return corrected
    elif iterations > 1:
        return remove_peak(corrected, window, iterations - 1)
    else:
        raise ValueError(f"Iterations must be positive integer, not {iterations}.")


# objects for torchvision.transforms.Compose


class Despiking:
    def __init__(self, args):
        self.in_pipeline = args[0]
        self.iterations = args[1]
        self.window_size = args[2]
        self.threshold = args[3]

    def __str__(self) -> str:
        return f"Despiking - iterations={self.iterations}, window_size={self.window_size}, threshold={self.threshold}"

    def __call__(self, spectrum):
        if self.in_pipeline:
            despiked = spectrum.copy()
            # despiking require only y axis
            despiked[1] = remove_peak(
                spectrum[1],
                iterations=self.iterations,
                window=self.window_size,
                threshold=self.threshold,
            )
            return despiked
        else:
            return spectrum
