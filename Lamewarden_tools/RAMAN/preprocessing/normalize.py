# -*- coding: utf-8 -*-
"""
normalize.py: 
@author: Ondrej Vaculik
@email: vacuon@isibrno.cz
"""

import numpy as np
from nptyping import NDArray
from sklearn.preprocessing import normalize

# objects for torchvision.transforms.Compose

class RamanNormalizer:
    """Normalize Raman sample using selected norm."""

    def __init__(self, settings):
        """
        Args:
            norm: {'l1', 'l2', 'max', 'minmax'}. Defaults to 'l2'.
        """
        try:
            self.norm = [key for key, value in settings.items() if value is True][0]
        except IndexError:
            self.norm = "no_norm"


    def __str__(self) -> str:
        return f"Normalized using {self.norm} norm"

    def __call__(self, spectrum: NDArray):
        normalized = spectrum.copy()
        if self.norm == "minmax":
            normalized[1] = (normalized[1] - np.min(normalized[1])) / (np.max(normalized[1]) - np.min(normalized[1]))
            return normalized
        elif self.norm == "no_norm":
            return spectrum
        else:
            # return only normalized spectrum X, leave out norms
            normalized[1] = normalize(np.array(spectrum[1]).reshape(1, -1), norm=self.norm)[0]
            return normalized # type: ignore
