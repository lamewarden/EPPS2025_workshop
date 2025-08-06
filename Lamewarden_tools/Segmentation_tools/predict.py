from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch.nn.functional as F
import cv2
import numpy as np
import torch
from torch.nn.functional import softmax
# from torch.utils.data import DataLoader
from Lamewarden_tools.Segmentation_tools.unet_tools import *



class UnetPredictor:
    def __init__(self, model_path, image, bboxes, patch_size=572, max_workers=12):

        valid_sizes = get_valid_patch_sizes()
        assert patch_size in valid_sizes, (f'Specified patch size of {patch_size}'
                f'is not valid. Valid patch sizes are {valid_sizes}')
        self.in_w = patch_size
        self.out_w = self.in_w - 72                
        self.image = image
        self.bboxes = bboxes
        mem_per_item = 3800000000
        total_mem = 0
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total_mem += torch.cuda.get_device_properties(i).total_memory
            self.bs = total_mem // mem_per_item
            self.bs = min(max_workers, self.bs)
        else:
            self.bs = 1 # cpu is batch size of 1
        self.model_path = model_path

    def predict(self):
        masks_dict = {}
        for name, box in self.bboxes.items():
            cut_image = self.image[box[1]:box[3], box[0]:box[2]]
            mask = ensemble_segment([self.model_path], cut_image, self.bs, self.in_w, self.out_w)
            mask_array = np.zeros(self.image.shape[:2], dtype=np.uint8)
            mask_array[box[1]:box[3], box[0]:box[2]] = mask
            masks_dict[name] = mask_array
        return masks_dict


def SAM_predict(model_path, image, bboxes):
    sam = sam_model_registry["vit_h"](checkpoint=model_path)
    sam.to(device='cuda')
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    masks_dict = {}
    for name, bbox in bboxes.items():
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox[None, :],
            multimask_output=False,
        )
        masks_dict[name]=masks[0].astype(int)
    return masks_dict
    
    
def Yolo_predict(model_path, image, bboxes):
    model = YOLO(model_path, task='segment')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    masks_dict = {}
    for name, box in bboxes.items():
        cut_image = image[box[1]:box[3], box[0]:box[2]]
        results = model(cut_image)
        for result in results:
            mask_significance = 0
            try:
                for mask in result.masks.data:
                    if mask.sum() > mask_significance:
                        mask_significance = mask.sum()
                        biggest_mask = mask.unsqueeze(0).unsqueeze(0)
                resized_mask = F.interpolate(biggest_mask, size=result.orig_shape, mode='bilinear', align_corners=False)
                resized_mask = resized_mask.squeeze(0).squeeze(0)
                binary_mask_2d = resized_mask.cpu().numpy().astype(int)
                mask_array = np.zeros(image.shape[:2], dtype=np.uint8)
                mask_array[box[1]:box[3], box[0]:box[2]] = binary_mask_2d
            except Exception as e:
                print("No plants were detected on the image.")
                mask_array = np.zeros(image.shape[:2], dtype=np.uint8)
                pass
            # to-do - add empty pixels around to create mask in the context of the original image
        masks_dict[name] = mask_array
    return masks_dict