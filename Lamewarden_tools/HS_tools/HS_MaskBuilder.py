import numpy as np
from skimage.morphology import binary_erosion, remove_small_objects
from Lamewarden_tools.HS_tools.readHS import *
from Lamewarden_tools.masks_convertor import *
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture

class HS_MaskBuilder:
    def __init__(self, hs_image, model):
        self.hs_image = hs_image
        self.model = model
        self.original_shape = hs_image.img.shape
        self.flattened_img = hs_image.img.reshape(-1, self.original_shape[2])
        
    def check_compatibility(self):
        if self.flattened_img.shape[1] != self.model.n_features_in_:
            raise ValueError("The dimensions of the HS_image data do not match the model's expected input dimensions.")
    
    def predict_probabilities(self):
        self.check_compatibility()
        probabilities = self.model.predict_proba(self.flattened_img)
        return probabilities
    
    def create_binary_mask(self, probabilities, threshold=0.5):
        binary_mask = (probabilities[:, 1] >= threshold).astype(np.uint8)
        binary_mask_2d = binary_mask.reshape(self.original_shape[0], self.original_shape[1])
        return binary_mask_2d
    
    def process_mask(self, binary_mask, erosion=None, remove_small=None):
        if erosion is not None:
            binary_mask = binary_erosion(binary_mask, footprint=np.ones((erosion, erosion)))
        if remove_small is not None:
            binary_mask = remove_small_objects(binary_mask.astype(bool), min_size=remove_small).astype(np.uint8)
        return binary_mask


import matplotlib.pyplot as plt

class HS_ThresholdOptimizer:
    def __init__(self, mask_builder):
        self.mask_builder = mask_builder
    
    def optimize_threshold(self, step=0.05):
        thresholds = np.arange(0, 1.05, step)
        probabilities = self.mask_builder.predict_probabilities()
        original_shape = self.mask_builder.original_shape
        
        fig, axes = plt.subplots(nrows=len(thresholds)//2 + len(thresholds)%2, ncols=2, figsize=(15, 5*len(thresholds)//2 + len(thresholds)%2))

        for i, threshold in enumerate(thresholds):
            binary_mask = self.mask_builder.create_binary_mask(probabilities, threshold=threshold)
            rgb_image = get_rgb_sample(self.mask_builder.hs_image.img, normalize=False, colorize_array=True)
            overlayed_image = overlay_image_with_mask(rgb_image, binary_mask, mask_color=(255, 0, 0), alpha=0.5)

            ax = axes.flatten()[i]
            ax.imshow(overlayed_image)
            ax.set_title(f"Threshold: {threshold:.2f}")
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

def split_mask_by_illumination(HS_image, mask, split_parameter = "std", threshold_search='otsu', cleaning_overlap_quantile=None, plot_data=False):
    # What are we taking as a parameter?
    if split_parameter == "std":
        X = np.std(HS_image.img, axis=2)
    elif split_parameter == "mean":
        X = np.mean(HS_image.img, axis=2)
    elif split_parameter == "max":
        X = np.max(HS_image.img, axis=2)
    elif int(split_parameter) in HS_image.ind:
        X = HS_image[int(split_parameter)].squeeze()
    else:
        raise ValueError("split_parameter must be 'std', 'mean', 'max' or a valid spectral band index")

    # extract the nonzero pixels from the mask and the image data, saving original mask shape for further recosntruction
    X_masked = (X*mask.squeeze())
    original_shape = X_masked.shape
    nonzero_mask = (X_masked != 0)
    X_reshaped = X_masked[nonzero_mask].astype(np.float64)
    
    # finding a treshold:
    if threshold_search == 'otsu':
    # Compute Otsu threshold
        threshold_val = threshold_otsu(X_reshaped)
    elif threshold_search == 'gmm':
     # Compute GMM threshold
        X_reshaped_for_gmm = X_reshaped.reshape(-1, 1)
        gmm = GaussianMixture(
            n_components=2,
            random_state=42,
            init_params='k-means++', 
            n_init=1  # Reduce number of initializations
        )
        gmm.fit(X_reshaped_for_gmm)
        labels_nonzero = gmm.predict(X_reshaped_for_gmm)
        gmm_df = pd.DataFrame({"values":X_reshaped_for_gmm.squeeze(), "label":labels_nonzero})
        threshold_val = np.max(gmm_df.groupby("label")["values"].apply(np.min).reset_index()["values"])
        print("GMM threshold: ", threshold_val)

    # Build the base shadow/illumination masks on the full array
    ill_mask = (X_reshaped > threshold_val)        # True = Shadow
    shadow_mask   = (X_reshaped <= threshold_val)       # True = Illumination
    if cleaning_overlap_quantile is not None:
    # "Cleanse" extremes based on quantiles
        q0_shadow = np.quantile(X_reshaped[shadow_mask], cleaning_overlap_quantile) if np.any(shadow_mask) else None
        q1_illum  = np.quantile(X_reshaped[ill_mask],  1-cleaning_overlap_quantile)  if np.any(ill_mask)  else None

        # Combine the conditions to keep the same total length as X_reshaped
        shadow_mask = shadow_mask & (X_reshaped < q0_shadow)
        ill_mask    = ill_mask &  (X_reshaped > q1_illum)

    # Now create 2D masks
    shadow_mask_2d = np.zeros(original_shape, dtype=bool)
    ill_mask_2d    = np.zeros(original_shape, dtype=bool)

    shadow_mask_2d[nonzero_mask] = shadow_mask
    ill_mask_2d[nonzero_mask]    = ill_mask

    # If you want a 2-channel mask array
    mask = np.zeros((*original_shape, 2), dtype=np.float32)
    mask[shadow_mask_2d, 0] = 1.0
    mask[ill_mask_2d,   1] = 1.0


    # Plot the results if requested
    if plot_data:
        df_list = []
        df_list.append(
            pd.DataFrame({
                "values": X_reshaped[ill_mask],
                "label":  0
            })
        )
        df_list.append(
            pd.DataFrame({
                "values": X_reshaped[shadow_mask],
                "label":  1
            })
        )

        df = pd.concat(df_list).reset_index(drop=True)

        plt.figure(figsize=(10,6))
        sns.histplot(data=df, x="values", hue="label", bins=100, alpha=0.5)
        plt.title("Cleaned Shadow vs. Illumination Intensity Distribution")
        plt.show()
        return mask
    else:
        return mask
