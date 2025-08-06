# -*- coding: utf-8 -*-
"""
filtering.py:
@author: Ondrej Vaculik
@email: vacuon@isibrno.cz
"""

from scipy.signal import savgol_filter
from .exceptions import SGFilterError

# objects for torchvision.transforms.Compose


class SGFilter(object):
    """Savitzky Golay spectrum filtering.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    """

    def __init__(self, settings):
        self.in_pipeline = settings[0]
        self.window_length = settings[1]
        self.polyorder = settings[2]

    def __str__(self) -> str:
        return f"SGF - window={self.window_length}, order={self.polyorder}"

    def __call__(self, spectrum):
        if self.in_pipeline:
            filtered = spectrum.copy()
            try:
                filtered[1] = savgol_filter(spectrum[1], self.window_length, self.polyorder)
            except ValueError as ve:
                raise SGFilterError(f"Savitzky-Golay filter failed: {str(ve)}")
            return filtered
        else:
            return spectrum
