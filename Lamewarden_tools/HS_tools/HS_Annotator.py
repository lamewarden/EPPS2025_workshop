
from collections import defaultdict
import napari
from Lamewarden_tools.HS_tools.readHS import *

class HS_Annotator:
    def __init__(self):
        self.masks = defaultdict(dict)
        self.applied_masks = defaultdict(dict)

    @classmethod
    def open_anotation_window(cls, hsi_series, image_to_show = "both"):
        instance = cls()
        
        instance.viewer = napari.Viewer()
        for name in hsi_series.content.keys():

            if image_to_show == "content":
                instance.viewer.add_image(hsi_series.content[name].rgb_sample, name=f'{name}')
            elif image_to_show == "PCA":
                instance.viewer.add_image(hsi_series.content[name].rgb_sample, name=f'{name}')
                instance.viewer.add_image(get_rgb_sample(hsi_series.content[name],normalize=False, colorize_array=True), name=f'colorized_{name}')

            instance.viewer.add_labels(np.zeros_like(hsi_series.content[name].img[:,:,-1]).astype('uint8'), name=f'BG_{name}')
            instance.viewer.add_labels(np.zeros_like(hsi_series.content[name].img[:,:,-1]).astype('uint8'), name=f'PT_{name}')
        napari.run()
        return instance

    def get_masks(self):
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels) and layer.data.sum()>0:
                main_key = layer.name[3:]
                sub_key = layer.name[:2]
                self.masks[main_key][sub_key] = layer.data

    
    def apply_mask(self, hsi_series,image_to_apply="content"):
        for name in self.masks.keys():
            print(name)
            if image_to_apply == "content":
                self.applied_masks[name]["PT"] = hsi_series.content[name].img * self.masks[name]["PT"][:, :, np.newaxis]
                self.applied_masks[name]["BG"] = hsi_series.content[name].img * self.masks[name]["BG"][:, :, np.newaxis]
            elif image_to_apply == "norm_content":
                self.applied_masks[name]["PT"] = hsi_series.content[name].norm_img * self.masks[name]["PT"][:, :, np.newaxis]
                self.applied_masks[name]["BG"] = hsi_series.content[name].norm_img * self.masks[name]["BG"][:, :, np.newaxis]


    def flatten_applied_mask(self, column_names):
        masked_df_list = []
        for name in self.applied_masks.keys():
            for key in self.applied_masks[name].keys():
                masked_pixels = self.applied_masks[name][key]

                masked_df = pd.DataFrame(masked_pixels[masked_pixels.mean(axis=2) != 0], columns=column_names)
                if key == "PT":
                    masked_df['label'] = 1
                elif key == "BG":
                    masked_df['label'] = 0
                masked_df['name'] = name
                masked_df_list.append(masked_df)
        return pd.concat(masked_df_list, axis=0, ignore_index=True)
    

