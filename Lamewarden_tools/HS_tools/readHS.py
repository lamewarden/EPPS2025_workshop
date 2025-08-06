
import numpy as np
import bisect
import spectral as sp
import pandas as pd
from Lamewarden_tools.HS_tools.convert_to_envi import * # just put script I sent you in same folder as this notebook
import matplotlib.pyplot as plt
import cv2
import math
# import napari
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import copy
import warnings
from scipy.ndimage import gaussian_filter, binary_erosion, median_filter
import seaborn as sns


# Ignore RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


# some primitive class to read data
class HS_image:
    epsilon = 1e-7
    def __init__(self, data_path):
        self.data_path = data_path
        self.read_hdr(data_path)


    def read_hdr(self, data_path):
        try:
            hdr = sp.open_image(data_path)
        except:
            convert_header_to_envi(data_path)
            hdr = sp.envi.open(data_path)
        self.hdr = hdr.bands.centers
        self.rows, self.cols, self.bands = hdr.nrows, hdr.ncols, hdr.nbands
        self.meta = hdr.metadata
        self.img = hdr.load()
        self.name = os.path.basename(data_path)
        self.ind = [int(float(x)) for x in self.meta['wavelength'][:-1]]
        self.bits = int(self.meta['data type'])
        self.normalized = False
        # self.bits = hdr.bits
        # self.line = self.name.split('-')[2]

    def __str__(self):
        return str(self.name)

    def calibrate(self, dc=False, clip_to=3):
        white_matrix, dark_matrix = self.upload_calibration(dc)
        # limiting the height of the reflectance
        self.img = np.clip((self.img - dark_matrix) / (white_matrix - dark_matrix), 0, clip_to)
        # filling nans with 0
        self.img[np.isnan(self.img)] = 0
        self.img[np.isinf(self.img)] = clip_to
        self.rgb_sample = get_rgb_sample(self, normalize=True)


    def upload_calibration(self, dc):
        # upload wc
        white_calibration = HS_image(os.path.join(self.data_path[:-8] + "WhiteCalibration.hdr"))
        white_matrix = np.mean(white_calibration.img, axis=0)
        if dc:
            dark_calibration = HS_image(os.path.join(self.data_path[:-8] + "DarkCalibration.hdr"))
            dark_matrix = np.mean(dark_calibration.img, axis=0)
        else:
            dark_matrix = np.zeros_like(white_matrix)
        return white_matrix, dark_matrix


    def apply_snv(self):
        """
        Applies SNV transformation to the entire hyperspectral image in a vectorized manner.
        """
        # Reshape the image to (num_pixels, num_bands)
        flat_img = self.img.reshape(-1, self.img.shape[-1])
        
        # Calculate mean and std along the spectral axis
        mean_spectrum = np.mean(flat_img, axis=1, keepdims=True)
        std_spectrum = np.std(flat_img, axis=1, keepdims=True)
        
        # Apply SNV and handle zero std deviation
        snv_image = (flat_img - mean_spectrum) / (std_spectrum + self.epsilon)  # Add epsilon to avoid division by zero
        
        # Reshape back to the original image dimensions
        self.img = snv_image.reshape(self.img.shape)

    
    
    def apply_min_max_normalization(self):
        # Create a new MS_image object to store the result
        # result = copy.deepcopy(self.img)
        if self.normalized == True:
            print("HS image is already normalized. No new transformation will be performed")
            pass
        # Apply min-max normalization to each channel
        for i in range(self.img.shape[2]):
            channel = self.img[:, :, i]
            min_val = np.min(channel)
            max_val = np.max(channel)
            self.img[:, :, i] = (channel - min_val) / (max_val - min_val)

    
    def apply_standard_scaler(self):
        flat_img = self.img.reshape(-1, self.img.shape[-1])
        scaler = StandardScaler()
        scaled_img = scaler.fit_transform(flat_img)
        self.img = scaled_img.reshape(self.img.shape)



    def normalize(self, to_wl = 1000, clip_to = 3):
        if self.normalized == True:
            print("HS image is already normalized. No new transformation will be performed")
        else:
            try:
                self.img = self.img / self[to_wl][:,:,np.newaxis]
            except ValueError:
                self.img = self.img / self[to_wl]
            self.img[np.isnan(self.img)] = 0
            self.img = np.clip(self.img, 0, clip_to)
            # finite_max = np.max(self.img[np.isfinite(self.img)])
            self.img[np.isinf(self.img)] = 0
            self.normalized=True

    def standardize(self):
        pass


    def apply_median_filter(self, size=3):
        filtered_image = np.zeros_like(self.img)
        for i in range(self.img.shape[2]):
            filtered_image[:, :, i] = median_filter(self.img[:, :, i], size=size)
        self.img = filtered_image
        return self.img


    def apply_gaussian_filter(self, sigma=3):
        filtered_image = np.zeros_like(self.img)
        for i in range(self.img.shape[2]):
            filtered_image[:, :, i] = gaussian_filter(self.img[:, :, i], sigma=sigma)
        return filtered_image

    def get_closest_wavelength(self, wl):
        pos = bisect.bisect_left(self.ind, wl)
        if pos == len(self.ind):
            return self.ind[-1]
        if pos == 0:
            return self.ind[0]
        if self.ind[pos] == wl:
            return self.ind[pos]
        return self.ind[pos]


    def __getitem__(self, wl):
        if isinstance(wl, slice):
            # Handle slice of wavelengths
            start, stop, step = wl.start, wl.stop, wl.step
            start = int(float(start)) if start is not None else self.ind[0]
            stop = int(float(stop)) if stop is not None else self.ind[-1]
            step = int(float(step)) if step is not None else 1
            
            start_index = self.ind.index(start)
            stop_index = self.ind.index(stop)
            step = step if step > 0 else 1

            return self.img[:, :, start_index:stop_index:step]
        else:
            # Handle single wavelength 
            wl = int(float(wl))
            closest_wl = self.get_closest_wavelength(wl)
            wl_index = self.ind.index(closest_wl)
            return self.img[:, :, wl_index]
    
        
    def __setitem__(self, wl, value):
        wl = int(float(wl))
        try:
            wl_index = self.ind.index(wl)
            self.img[:,:,wl_index] = value
        except:
            raise IndexError("Entered wavelength is not in spectrum")
        
    @staticmethod
    def divide_arrays(array_3d, array_other, remove_outliers=False, sigma_threshold=2):
        """
        Divide a 3D array by another array (3D or 2D), handling divisions by zero and removing outliers.

        Parameters:
        - array_3d: np.ndarray
            The 3D array to be divided.
        - array_other: np.ndarray
            The array to divide by, can be 3D or 2D.
        - remove_outliers: bool
            Whether to remove outliers based on sigma threshold.
        - sigma_threshold: float
            The number of standard deviations to use as the threshold for outlier removal.

        Returns:
        - np.ndarray
            The resulting 3D array after division and optional outlier removal.
        """
        # Ensure input is a numpy array
        array_3d = np.asarray(array_3d)
        array_other = np.asarray(array_other)
        
        # Check the shape compatibility for broadcasting
        if array_other.ndim == 2:
            if array_3d.shape[:2] != array_other.shape:
                raise ValueError("The 2D array must have the same shape as the first two dimensions of the 3D array.")
            # Expand dimensions to make broadcasting work
            array_other = array_other[:, :, np.newaxis]
        elif array_other.ndim == 3 and array_other.shape[-1] == 1:
            if array_3d.shape[:2] != array_other.shape[:2]:
                raise ValueError("The 3D array with shape (X, Y, 1) must match the first two dimensions of the 3D array.")
        
        # Perform division with handling for divide by zero
        result = np.divide(array_3d, array_other, where=array_other!=0, out=np.zeros_like(array_3d))
        
        # Replace Inf values with 0 (since np.divide can create Inf)
        result[np.isinf(result)] = 0

        if remove_outliers:
            # Remove outliers along the third axis for each slice [:,:,i]
            for i in range(result.shape[2]):
                slice_ = result[:, :, i]
                mean = np.mean(slice_)
                std = np.std(slice_)
                # Identify outliers
                outliers = np.abs(slice_ - mean) > sigma_threshold * std
                # Set outliers to zero
                slice_[outliers] = 0
                # Assign back the slice with removed outliers
                result[:, :, i] = slice_
        
        return result
        
    @staticmethod
    def img_align(img, template, inplace = False):
        # trim 100 pixels from each side of the image
        # img_raw = copy.deepcopy(img)
        # template_raw = copy.deepcopy(template)
        # img.body = img.body[100:-100, 100:-100]
        # template.body = template.body[100:-100, 100:-100]
        sift = cv2.SIFT_create()
        template_bw = cv2.normalize(template, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img_bw = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        kp_temp, des_temp = sift.detectAndCompute(template_bw, None)
        kp_ch, des_ch = sift.detectAndCompute(img_bw, None)
        # Match descriptors using FLANN (or you can use BFMatcher for brute force matching)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des_ch, des_temp, k=2)
        # Store the good matches using Lowe's ratio test
        good_matches = []
        for m,n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        height, width, *_ = template.shape

        # Ensure we have enough matches to compute the homography
        if len(good_matches) > 10:
            src_pts = np.float32([kp_ch[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
            dst_pts = np.float32([kp_temp[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

            # Compute the homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Warp the image
            if inplace is True:
                img = cv2.warpPerspective(img, M, (width, height))

            return M, width, height
        # if not enough matches found - return M as an identity matrix
        return np.eye(3), width, height
    

    def flatten_to_df(self):
        """
        Flattenting whole HS image into 2D DF with separate pixels as rows and WL as columns.
        
        """               
        return pd.DataFrame(self.img[self.img.mean(axis=2) != 0], columns=self.ind)


class MS_image(HS_image):
    def __init__(self, data_path=None):
        super().__init__(data_path)
        self.devignet_counter = 0


    def devignet(self, ref_HS, sigma=10, deblack=False, black_noise=0.0586):
    # Extracting de-vignetting matrix from ref images:
    # Gaussian blur application
        if self.devignet_counter == 1:
            return
        
        for channel in self.ind:

            # Gaussian blur
            ref_frame = cv2.GaussianBlur(ref_HS[channel], (3, 3), sigmaX=sigma)
            # Inverting
            ref_frame = np.round(1 / ref_frame, 9)
            # Dividing by mean
            ref_frame = (ref_frame / np.mean(ref_frame))[:,:,np.newaxis]

            # Applying calculated de-vignetting mask for the every image in the original TS
            if deblack:
                self[channel] = np.clip((self[channel] * ref_frame - black_noise * 4095), 0, 4095).astype(np.uint16)
            else:
                self[channel] = np.clip((self[channel] * ref_frame), 0, 4095).astype(np.uint16)
        
        self.devignet_counter = 1
        return 
    
    # def devignet(self, ref_HS, sigma=10, deblack=False, black_noise=0.0586):
    # # Extracting de-vignetting matrix from ref images:
    # # Gaussian blur application
    #     if self.devignet_counter == 1:
    #         return
        
    #     for channel in self.ind:

    #         # Gaussian blur
    #         ref_frame = cv2.GaussianBlur(ref_HS[channel], (3, 3), sigmaX=sigma)
    #         # Inverting
    #         ref_frame = np.round(1 / ref_frame, 9)
    #         # Dividing by mean
    #         ref_frame = (ref_frame / np.mean(ref_frame))[:,:,np.newaxis]

    #         # Applying calculated de-vignetting mask for the every image in the original TS
    #         if deblack:
    #             self[channel] = np.clip((self[channel] * ref_frame - black_noise * 4095), 0, 4095).astype(np.uint16)
    #         else:
    #             self[channel] = np.clip((self[channel] * ref_frame), 0, 4095).astype(np.uint16)
        
    #     self.devignet_counter = 1
    #     return 

    
    def devignet_old_school(self, ref_HS, sigma=10, deblack = False, black_noise = 0.0586):
        # extracting de-vignetting matrix from ref images:
        # gaussian blur application
        if self.devignet_counter == 1:
            return ref_HS
        
        ref_HS_copy = copy.deepcopy(ref_HS)
        # ref_list = []
        for channel in range(0,6):
            # gaussian blur
            ref_HS_copy.img[:,:,channel] = cv2.GaussianBlur(ref_HS_copy.img[:,:,channel] , (3, 3), sigmaX=sigma)
            # inverting 
            ref_HS_copy.img[:,:,channel]  = 1/ref_HS_copy.img[:,:,channel] 
            # dividing by mean
            ref_HS_copy.img[:,:,channel]  = ref_HS_copy.img[:,:,channel] /np.mean(ref_HS_copy.img[:,:,channel] )
            # ref_list.append(ref_HS_copy.img[channel])
        # Plying calculated de-vignetting mask for the every image in the original TS:
            if deblack is True:
                self.img[:,:,channel]  = np.clip((self.img[:,:,channel]  * ref_HS_copy.img[:,:,channel][:,:,np.newaxis] - 4095*black_noise), 0, 4095).astype(np.uint16) 
            else:
                self.img[:,:,channel] = np.clip((self.img[:,:,channel] * ref_HS_copy.img[:,:,channel][:,:,np.newaxis]), 0, 4095).astype(np.uint16)
        self.devignet_counter = 1


        return ref_HS_copy



# read data
def get_hdr_images(folder, min_rows = 1, format='hdr'):
    all_images = {}
    for root, dirs, files in os.walk(folder):
        for file in files:
            filepath = os.path.join(root, file)
            # select only files with .hdr extension and with more than 15 rows
            if filepath.endswith('.hdr') and int(HS_image(filepath).rows) > min_rows:
                if format == 'hdr':
                    img = HS_image(filepath)
                else:
                    img = MS_image(filepath)
                all_images[img.name] = img
    return all_images


def get_rgb_sample(image, normalize=True, colorize_array=False, correct=True):
    # For now we won't stretch all images, just one for visualization

    if colorize_array:
        R = image.img[:,:,0]
        G = image.img[:,:,1]
        B = image.img[:,:,2]
    elif len(image.ind) <= 6:
        R = (image[670] / 4095)
        G = (image[595] / 4095)
        B = (image[495] / 4095)
    elif np.mean(image.ind) < 900 and len(image.ind) > 6:
        R = np.mean([image[value] for value in image.ind if value >= 570 and value <= 650],axis=0)
        G = np.mean([image[value] for value in image.ind if value >= 520 and value <= 570],axis=0)
        B = np.mean([image[value] for value in image.ind if value >= 450 and value <= 520],axis=0)
    elif np.mean(image.ind) > 900:
        R = np.mean([image[value] for value in image.ind if value >= 1000 and value <= 1100],axis=0)
        G = np.mean([image[value] for value in image.ind if value >= 1200 and value <= 1300],axis=0)
        B = np.mean([image[value] for value in image.ind if value >= 1400 and value <= 1500],axis=0)
    if correct:
        # Remove outliers (values that are more than 3 standard deviations away from the mean)
        R = np.where(np.abs(R - np.mean(R)) > 2*np.std(R), np.mean(R), R)
        G = np.where(np.abs(G - np.mean(G)) > 2*np.std(G), np.mean(G), G)
        B = np.where(np.abs(B - np.mean(B)) > 2*np.std(B), np.mean(B), B)
        # Replace NaNs with the mean of the layer
        R = np.nan_to_num(R, nan=np.nanmin(R))
        G = np.nan_to_num(G, nan=np.nanmin(G))
        B = np.nan_to_num(B, nan=np.nanmin(B))

        # Replace infs with the mean of the layer
        R = np.where(np.isinf(R), np.nanmax(R), R)
        G = np.where(np.isinf(G), np.nanmax(G), G)
        B = np.where(np.isinf(B), np.nanmax(B), B)
    if normalize:
        R = R/np.max(R)
        G = G/np.max(G)
        B = B/np.max(B)
    # converting sample to the dict of spectral bands
    rgb_sample = np.dstack((R, G, B))
    return rgb_sample

def get_rgb_from_array(image, normalize=True, correct=True):
    # Calculate the size of each third of the layers
    fifth = image.shape[2] // 5

    # Extract the R, G, and B layers
    B = np.mean(image[:,:,50:fifth], axis=2)
    G = np.mean(image[:,:,2*fifth:3*fifth], axis=2)
    R = np.mean(image[:,:,3*fifth:4*fifth], axis=2)

    if correct:
        # Replace NaNs with the mean of the layer
        R = np.nan_to_num(R, nan=np.nanmin(R))
        G = np.nan_to_num(G, nan=np.nanmin(G))
        B = np.nan_to_num(B, nan=np.nanmin(B))

        # Replace infs with the mean of the layer
        R = np.where(np.isinf(R), np.nanmax(R), R)
        G = np.where(np.isinf(G), np.nanmax(G), G)
        B = np.where(np.isinf(B), np.nanmax(B), B)

        # Remove outliers (values that are more than 3 standard deviations away from the mean)
        R = np.where(np.abs(R - np.mean(R)) > 3*np.std(R), np.mean(R), R)
        G = np.where(np.abs(G - np.mean(G)) > 3*np.std(G), np.mean(G), G)
        B = np.where(np.abs(B - np.mean(B)) > 3*np.std(B), np.mean(B), B)

    if normalize:
        R = R/np.max(R)
        G = G/np.max(G)
        B = B/np.max(B)

    # Stack the R, G, and B layers to form the RGB image
    rgb_sample = np.dstack((B, G, R))

    return rgb_sample


def stretchImage(image, to_size=400, axis=0):
 
    return np.repeat(image.img, to_size//image.img.shape[axis], axis=axis)
    

def img_align(img, template):
    
    """
    Aligns an image to a template using feature matching and homography.

    Parameters:
    - img (cv2.Mat): The target image to be aligned. Expected to have attribute `body` which is a numpy array.
    - template (cv2.Mat): The template image. Expected to have attribute `body` which is a numpy array.

    Returns:
    - np.array: The homography matrix (3x3) that aligns the target image to the template if enough matches are found. None otherwise.

    The function trims 100 pixels from each side of both images, computes key points and descriptors using SIFT, matches descriptors using FLANN,
    filters good matches using Lowe's ratio test, computes a homography matrix if enough good matches are found, and applies this matrix to 
    warp the target image to align it with the template.
    """

    # trim 100 pixels from each side of the image
    # img_raw = copy.deepcopy(img)
    # template_raw = copy.deepcopy(template)
    # img.body = img.body[100:-100, 100:-100]
    # template.body = template.body[100:-100, 100:-100]
    sift = cv2.SIFT_create()
    kp_temp, des_temp = sift.detectAndCompute(template.img, None)
    kp_ch, des_ch = sift.detectAndCompute(img.img, None)
    # Match descriptors using FLANN (or you can use BFMatcher for brute force matching)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des_ch, des_temp, k=2)
    # Store the good matches using Lowe's ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Ensure we have enough matches to compute the homography
    if len(good_matches) > 10:
        src_pts = np.float32([kp_ch[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp_temp[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        # Compute the homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp the image
        height, width = template.img.shape
        img.img = cv2.warpPerspective(img.img, M, (width, height))
        return M
    # if no homografy found
    print("Not enough matches for homography found.")
    return None


def NDVICalculator(HS_image):
    NIR = np.mean(HS_image[725:785], axis=2)
    RED = np.mean(HS_image[660:670], axis=2)
    NDVI = np.clip((NIR - RED) / ((NIR + RED)), -1,1)
    NDVI[np.isnan(NDVI)] = 0
    return NDVI

def NDVIPlotter(NDVI_image, title="NDVI"):
    plt.imshow(NDVI_image, cmap='RdYlGn')  # 'RdYlGn' colormap is often used for NDVI
    plt.colorbar()  # Add a colorbar to show the color scale
    plt.title(title)
    plt.axis('off')
    plt.show()

def get_polygon_masks_from_json(json_file_path):
    """
    Converts polygon annotations from a Labelme JSON file into binary masks.
    
    Args:
        json_file_path (str): Path to the Labelme annotation JSON file.
        
    Returns:
        dict: A dictionary where keys are labels (e.g., "rotten") and values are binary masks (numpy arrays).
    """
    # Load the annotation JSON file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Get the image dimensions from the JSON data
    image_width = data.get('imageWidth')
    image_height = data.get('imageHeight')
    
    # Initialize a dictionary to store the binary masks for each label
    masks = {}
    
    # Iterate over each shape in the JSON file
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            label = shape['label']  # Get the label (e.g., "rotten", "fresh")
            
            # Initialize a mask for the label if not already created
            if label not in masks:
                masks[label] = np.zeros((image_height, image_width), dtype=np.uint8)
            
            # Get the polygon points
            polygon = np.array(shape['points'], dtype=np.int32)
            
            # Draw the polygon on the corresponding label's mask
            cv2.fillPoly(masks[label], [polygon], color=1)
    
    return masks


def standardize_image(image: np.ndarray) -> np.ndarray:
    """
    Standardizes a 3D image array (Height x Width x Channels).

    Parameters:
    image (np.ndarray): The input image in the form of a 3D array.
    
    Returns:
    np.ndarray: The standardized image.
    """
    # Ensure the image is in float format
    image = image.astype(np.float32)

    # Calculate mean and standard deviation for each channel
    means = np.mean(image, axis=(0, 1), keepdims=True)
    stds = np.std(image, axis=(0, 1), keepdims=True)

    # Avoid division by zero by setting stds to 1 where std is zero
    stds[stds == 0] = 1

    # Standardize each channel: (value - mean) / std
    standardized_image = (image - means) / stds

    return standardized_image

class Preprocessor:
    def __init__(self):
        self.operations = []

    def apply_operation(self, hs_image, operation, *args, **kwargs):
        # Apply the operation
        method = getattr(hs_image, operation)
        result = method(*args, **kwargs)
        
        # Record the operation and its arguments
        self.operations.append((operation, args, kwargs))
        return result

    def reapply_operations(self, hs_image):
        for operation, args, kwargs in self.operations:
            method = getattr(hs_image, operation)
            method(*args, **kwargs)

    def save_operations(self, file_path):
        np.save(file_path, self.operations)

    def load_operations(self, file_path):
        self.operations = np.load(file_path, allow_pickle=True).tolist()


class HS_image_transformer: # all transformation methods should be placed here
    pass

class HS_image_analyzer: # all analyzis methods should be placed here
    pass