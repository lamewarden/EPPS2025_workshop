import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.linear_model import RANSACRegressor, LinearRegression, TheilSenRegressor
import statsmodels.api as sm
import copy
import matplotlib.pyplot as plt
class HS_processor:
    """
    A class for processing hyperspectral (HS) image data, focusing on spectral smoothing and outlier removal.

    Provides methods for:

    - **Spectral Filtering:** Applying median, Gaussian, Savitzky-Golay, high-pass, and FFT-based low-pass/high-pass filters along the spectral dimension.
    - **Robust Regression:** Fitting polynomials to each spectrum using RANSAC, Theil-Sen, and Iteratively Re-weighted Least Squares (IRLS) for robust outlier handling.
    - **Combined Smoothing:**  A pipeline combining median filtering, RANSAC polynomial fitting, and optional Gaussian smoothing.
    - **Visualization:** Plotting the spectrum of a specified pixel.
    - **Method Chaining:** Allows applying a sequence of processing steps in a fluent style (requires an HSImage class with an .img attribute).

    The class operates on 3D numpy arrays representing hyperspectral data (height, width, spectral_bands).  Most methods process the data *spectrally*, meaning they apply the operation independently to each pixel's spectrum.

    Example Usage (assuming you have a 3D numpy array 'hs_data'):

        processor = HS_processor()

        # Apply a median filter:
        filtered_data = processor.median_filter_spectral(hs_data, window_size=5)

        # Apply RANSAC polynomial fitting:
        ransac_data = processor.ransac_polyfit_spectral(hs_data, degree=3)

        # Combined smoothing:
        smoothed_data = processor.robust_spectral_smoothing(hs_data, median_window=3, poly_degree=5, final_sigma=2)

        # Plot a spectrum (assuming pixel coordinates (10, 20)):
        processor.plot_spectrum(hs_data, (10, 20))


    Method Chaining Example (assuming an HSImage class):

        # Assuming you have an HSImage object 'hs_image'
        processed_image = (processor
                           .median_filter_spectral_chain(hs_image, window_size=3)
                           .ransac_polyfit_spectral_chain(degree=2)
                           )

    """

    def __init__(self):
        """
        Initializes the HS_processor.  Currently, no initialization parameters are used.
        """
        pass

    def median_filter_spectral(self, data, window_size=3):
        """
        Applies a median filter along the spectral dimension of the hyperspectral data.

        Args:
            data (numpy.ndarray): The input hyperspectral data as a 3D numpy array (height, width, spectral_bands).
            window_size (int): The size of the median filter window. Must be an odd integer.

        Returns:
            numpy.ndarray: The filtered data with the same shape as the input.

        Raises:
            ValueError: If the window_size is not an odd integer.
        """
        if window_size % 2 == 0:
            raise ValueError("Window size must be odd.")

        filtered_data = median_filter(data, size=(1, 1, window_size))
        return filtered_data

    def gaussian_spectral(self, data, sigma=3):
        """
        Applies a Gaussian filter along the spectral dimension.

        Args:
            data (numpy.ndarray): The input hyperspectral data (height, width, spectral_bands).
            sigma (float): The standard deviation of the Gaussian kernel.

        Returns:
            numpy.ndarray: The filtered data.
        """
        filtered_data = gaussian_filter1d(data, sigma=sigma, axis=2)
        return filtered_data

    def savgol_spectral(self, data, window_size=15, polyorder=3):
        """
        Applies a Savitzky-Golay filter along the spectral dimension.

        Args:
            data (numpy.ndarray): The input hyperspectral data (height, width, spectral_bands).
            window_size (int): The length of the filter window. Must be an odd integer.
            polyorder (int): The order of the polynomial used to fit the samples.

        Returns:
            numpy.ndarray: The filtered data.
        """
        filtered_data = np.apply_along_axis(
            lambda m: savgol_filter(m, window_size, polyorder),
            axis=2,
            arr=data
        )
        return filtered_data

    def highpass_spectral(self, data, sigma=3):
        """
        Applies a high-pass filter by subtracting a Gaussian-smoothed version.

        Args:
            data (numpy.ndarray):  The input hyperspectral data (height, width, spectral_bands).
            sigma (float):  The standard deviation of the Gaussian kernel used for smoothing.

        Returns:
            numpy.ndarray: The high-pass filtered data.
        """
        lowpass_data = self.gaussian_spectral(data, sigma=sigma)
        highpass_data = data - lowpass_data
        return highpass_data

    def fft_lowpass_spectral(self, data, cutoff_fraction=0.1):
        """
        Applies a low-pass filter in the frequency domain using the Fast Fourier Transform (FFT).

        Args:
            data (numpy.ndarray): The input hyperspectral data (height, width, spectral_bands).
            cutoff_fraction (float): The fraction of frequencies to keep (0 to 1).  Values closer to 0 remove more high-frequency components.

        Returns:
            numpy.ndarray: The low-pass filtered data.
        """
        fft_data = np.fft.fft(data, axis=2)
        num_bands = data.shape[2]
        cutoff_index = int(num_bands * cutoff_fraction)
        mask = np.zeros(num_bands)
        mask[:cutoff_index] = 1
        filtered_fft_data = fft_data * mask
        filtered_data = np.fft.ifft(filtered_fft_data, axis=2).real
        return filtered_data

    def fft_highpass_spectral(self, data, cutoff_fraction=0.1):
        """
        Applies a high-pass filter in the frequency domain using the Fast Fourier Transform (FFT).

        Args:
            data (numpy.ndarray): The input hyperspectral data (height, width, spectral_bands).
            cutoff_fraction (float):  The fraction of frequencies to remove (0 to 1). Values closer to 0 remove more low-frequency components.

        Returns:
            numpy.ndarray: The high-pass filtered data.
        """
        fft_data = np.fft.fft(data, axis=2)
        num_bands = data.shape[2]
        cutoff_index = int(num_bands * cutoff_fraction)
        mask = np.ones(num_bands)
        mask[:cutoff_index] = 0
        filtered_fft_data = fft_data * mask
        filtered_data = np.fft.ifft(filtered_fft_data, axis=2).real
        return filtered_data

    def ransac_polyfit_spectral(self, data, degree=3, **kwargs):
        """
        Fits a polynomial to each spectrum using RANSAC (RANdom SAmple Consensus) for robust outlier rejection.

        Args:
            data (numpy.ndarray): The input hyperspectral data (height, width, spectral_bands).
            degree (int): The degree of the polynomial to fit.
            **kwargs:  Additional keyword arguments to pass to the RANSACRegressor (e.g., `min_samples`, `residual_threshold`, `max_trials`).

        Returns:
            numpy.ndarray: The data with each spectrum replaced by its RANSAC polynomial fit.
        """
        height, width, spectral_bands = data.shape
        filtered_data = np.zeros_like(data)
        x = np.arange(spectral_bands)

        for h in range(height):
            for w in range(width):
                spectrum = data[h, w, :]
                X = x.reshape(-1, 1)
                y = spectrum.reshape(-1, 1)

                if np.all(y == 0) or np.all(y == y[0]):
                    filtered_data[h, w, :] = y.flatten()
                    continue
                residual_threshold = kwargs.pop('residual_threshold', 0.5)
                ransac = RANSACRegressor(LinearRegression(),
                                         min_samples=0.5,
                                         residual_threshold=residual_threshold,
                                         max_trials=100,
                                         **kwargs)
                try:
                    ransac.fit(X, y)
                    filtered_data[h, w, :] = ransac.predict(X).flatten()
                except ValueError:
                    filtered_data[h, w, :] = np.median(spectrum)
        return filtered_data

    def theilsen_polyfit_spectral(self, data, degree=3, **kwargs):
        """
        Fits a polynomial to each spectrum using Theil-Sen regression, then fits a polynomial of specified degree to the residuals.

        Args:
            data (numpy.ndarray): The input hyperspectral data (height, width, spectral_bands).
            degree (int): The degree of the polynomial to fit to the residuals after the initial linear Theil-Sen fit.
            **kwargs: Additional keyword arguments to be passed to TheilSenRegressor.

        Returns:
            numpy.ndarray:  The data with each spectrum replaced by the sum of the linear Theil-Sen fit and the polynomial fit to the residuals.
        """
        height, width, spectral_bands = data.shape
        filtered_data = np.zeros_like(data)
        x = np.arange(spectral_bands)

        for h in range(height):
            for w in range(width):
                spectrum = data[h, w, :]
                X = x.reshape(-1, 1)
                y = spectrum.reshape(-1, 1)

                if np.all(y == 0) or np.all(y == y[0]):
                    filtered_data[h, w, :] = y.flatten()
                    continue

                theil_sen = TheilSenRegressor(**kwargs)
                try:
                    theil_sen.fit(X, y.ravel())
                    linear_fit = theil_sen.predict(X)
                    detrended_spectrum = spectrum - linear_fit
                    poly_model = np.poly1d(np.polyfit(x, detrended_spectrum, degree))
                    poly_fit = poly_model(x)
                    filtered_data[h, w, :] = poly_fit + linear_fit
                except ValueError as e:
                    filtered_data[h, w, :] = np.median(spectrum)

        return filtered_data

    def irls_polyfit_spectral(self, data, degree=3):
        """
        Fits a polynomial to each spectrum using Iteratively Re-weighted Least Squares (IRLS).

        Args:
            data (numpy.ndarray): The input hyperspectral data (height, width, spectral_bands).
            degree (int): The degree of the polynomial to fit.

        Returns:
            numpy.ndarray: The data with each spectrum replaced by its IRLS polynomial fit.
        """
        height, width, spectral_bands = data.shape
        filtered_data = np.zeros_like(data)
        x = np.arange(spectral_bands)

        for h in range(height):
            for w in range(width):
                spectrum = data[h, w, :]
                X = sm.add_constant(np.vander(x, degree + 1))
                y = spectrum

                if np.all(y == 0) or np.all(y == y[0]):
                    filtered_data[h,w,:] = y
                    continue

                try:
                    rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
                    rlm_results = rlm_model.fit()
                    filtered_data[h, w, :] = rlm_results.fittedvalues
                except ValueError:
                    filtered_data[h,w,:] = np.median(spectrum)

        return filtered_data

    def robust_spectral_smoothing(self, data, median_window=3, poly_degree=5, ransac_threshold=0.5, final_sigma=None):
        """
        Combines median filtering, RANSAC polynomial fitting, and optional Gaussian smoothing.

        This method provides a robust smoothing pipeline:
        1. Applies a median filter to remove impulsive noise.
        2. Fits a polynomial to each spectrum using RANSAC to handle outliers.
        3. Optionally applies a final Gaussian smoothing step.

        Args:
            data (numpy.ndarray): The input hyperspectral data (height, width, spectral_bands).
            median_window (int): The size of the median filter window.
            poly_degree (int): The degree of the polynomial for RANSAC fitting.
            ransac_threshold (float): The residual threshold for RANSAC.
            final_sigma (float, optional): The standard deviation for the final Gaussian smoothing. If None, no Gaussian smoothing is applied.

        Returns:
            numpy.ndarray: The smoothed data.
        """
        cleaned_data = self.median_filter_spectral(data, window_size=median_window)
        ransac_data = self.ransac_polyfit_spectral(cleaned_data, degree=poly_degree, residual_threshold=ransac_threshold)
        if final_sigma is not None:
            final_data = self.gaussian_spectral(ransac_data, sigma=final_sigma)
        else:
            final_data = ransac_data
        return final_data

    def plot_spectrum(self, data, pixel_coords, title="Spectrum"):
        """
        Plots the spectrum of a single pixel.

        Args:
            data (numpy.ndarray): The hyperspectral data (height, width, spectral_bands).
            pixel_coords (tuple): The (x, y) coordinates of the pixel.  Note that x corresponds to the row (height) and y to the column (width).
            title (str): The title of the plot.

        """
        h, w = pixel_coords
        spectrum = data[h,w,:]

        plt.figure(figsize=(8, 6))
        plt.plot(spectrum)
        plt.xlabel("Spectral Band")
        plt.ylabel("Reflectance/Intensity")
        plt.title(title)
        plt.grid(True)
        plt.show()

    def median_filter_spectral_chain(self, hs_image, window_size=3):
        """
        Applies a median filter along the spectral dimension (chainable version).

        Args:
            hs_image: An object with a `.img` attribute representing the hyperspectral data.
            window_size (int): The size of the median filter window.

        Returns:
            A new object of the same type as `hs_image`, with the filtered data in its `.img` attribute.
        """
        filtered_data = self.median_filter_spectral(hs_image.img, window_size)
        new_hs_image = copy.deepcopy(hs_image)  # Important: Create a copy!
        new_hs_image.img = filtered_data
        return new_hs_image

    def ransac_polyfit_spectral_chain(self, hs_image, degree=3, **kwargs):
        """
        Applies RANSAC polynomial fitting (chainable version).

        Args:
            hs_image: An object with a `.img` attribute representing the hyperspectral data.
            degree (int): The degree of the polynomial.
            **kwargs: Keyword arguments for RANSACRegressor.

        Returns:
            A new object with the filtered data.
        """
        filtered_data = self.ransac_polyfit_spectral(hs_image.img, degree, **kwargs)
        new_hs_image = copy.deepcopy(hs_image)
        new_hs_image.img = filtered_data
        return new_hs_image

    def robust_spectral_smoothing_chain(self, hs_image, median_window=3, poly_degree=5, ransac_threshold=0.5, final_sigma=None):
        """
        Applies combined median filtering, RANSAC, and Gaussian smoothing (chainable version).

        Args:
            hs_image:  An object with a .img attribute representing the hyperspectral data.
            median_window (int): Median filter window size.
            poly_degree (int):  Polynomial degree for RANSAC.
            ransac_threshold (float): RANSAC residual threshold.
            final_sigma (float, optional):  Gaussian smoothing sigma.

        Returns:
            A new object with the smoothed data.
        """
        filtered_data = self.robust_spectral_smoothing(hs_image.img, median_window, poly_degree, ransac_threshold, final_sigma)
        new_hs_image = copy.deepcopy(hs_image)
        new_hs_image.img = filtered_data
        return new_hs_image