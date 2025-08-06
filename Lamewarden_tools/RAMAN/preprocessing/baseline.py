# -*- coding: utf-8 -*-
"""
baseline.py:
@author: Ondrej Vaculik
@email: vacuon@isibrno.cz
"""

from nptyping import NDArray
from pybaselines import Baseline
from skimage.restoration import rolling_ball

# functions


def rolling_circle(spectrum: NDArray, radius: int = 1000, passes: int = 10):
    """Rolling circle raman spectral filtering.
    Can be peroformed multiple times.
    https://www.researchgate.net/publication/7170761_Optimization_of_the_Rolling-Circle_Filter_for_Raman_Background_Subtraction

    Args:
        spectrum (NDArray): Spectral vector
        radius (int): Rolling circle radius
        passes (int, optional): Number of repetitions. Defaults to 1.

    Returns:
        NDArray: Spectrum with removed background.
    """
    for _ in range(passes):
        spectrum -= rolling_ball(spectrum, radius=radius)
    return spectrum


def polynomial(spectrum: NDArray, poly_order: int = 12):
    """imodpoly (Improved Modified Polynomial)
    https://pybaselines.readthedocs.io/en/latest/algorithms/polynomial.html#imodpoly-improved-modified-polynomial
    """
    baseline_fitter = Baseline()
    baseline = baseline_fitter.imodpoly(spectrum, poly_order)[0]
    spectrum -= baseline
    return spectrum


def snip(spectrum: NDArray):
    """snip (Statistics-sensitive Non-linear Iterative Peak-clipping)
    https://pybaselines.readthedocs.io/en/latest/algorithms/smooth.html#snip-statistics-sensitive-non-linear-iterative-peak-clipping
    """
    baseline_fitter = Baseline()
    baseline = baseline_fitter.snip(spectrum)[0]
    spectrum -= baseline
    return spectrum


# # objects for torchvision.transforms.Compose
# class RollingCircle(object):
#     """Remove fluorescence background using Rolling Circle algorithm.
#     https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.rolling_ball
#     """

#     def __init__(self, radius: int = 1000, passes: int = 10):
#         assert isinstance(radius, int) and radius > 1
#         assert isinstance(passes, int) and passes > 0
#         self.radius = radius
#         self.passes = passes

#     def __str__(self) -> str:
#         return f"RCF - radius={self.radius}, passes={self.passes}"

#     def __call__(self, spectrum: NDArray):
#         corrected = spectrum.copy()
#         corrected = rolling_circle(spectrum, self.radius, self.passes)
#         return corrected


class BaselineCorrection:
    """Remove fluorescence background using pybaseline library.
    https://pybaselines.readthedocs.io/en/latest/algorithms/smooth.html
    Available algorithms: 'snip', 'ipf', rolling ball
    """

    def __init__(self, settings):
        self.poly_order = settings['ipf'][1]
        self.radius = settings['rolling_ball'][1]
        self.passes = settings['rolling_ball'][2]
        try:
            self.algorithm = [key for key, value in settings.items() if value[0] is True][0]
        except IndexError:
            self.algorithm = "no_baseline"

    def __str__(self) -> str:
        return (
            f"BaselineCorrection - algorithm={self.algorithm}, order={self.poly_order}, radius={self.radius}, passes = {self.passes}"
        )

    def __call__(self, spectrum: NDArray):
        corrected = spectrum.copy()
        if self.algorithm =='rolling_ball':
            corrected[1] = rolling_circle(corrected[1], self.radius, self.passes)
            return corrected
        elif self.algorithm == "snip":
            corrected[1] = snip(corrected[1])
            return corrected
        elif self.algorithm == "ipf":
            corrected[1] = polynomial(corrected[1], self.poly_order)
            return corrected
        else:
            return spectrum

