import xml.etree.ElementTree as ET
import numpy as np
import cv2
import argparse
import torch
import os
# import shutil
import numpy as np
from scipy.ndimage import label, find_objects

class TrayMaskProcessor:
    def __init__(self) -> None:
        self.bboxes={}


    def mask_upload(self, xml_path):
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        self.width = int(self.root.attrib['width'])
        self.height = int(self.root.attrib['height'])
        self.mask = np.zeros((self.height, self.width), dtype=np.uint8)  # Black mask

    def extract_from_all(self):
        self.extract_from_circle()
        self.extract_from_rectangle()
        self.extract_from_polygon()
        self.extract_from_pie()
        return self.bboxes


    def extract_from_rectangle(self):
        for rectangle in self.root.findall('.//TRectangleShape'):
            # Get the coordinates and convert them to integer
            roi_name = str(rectangle.attrib['name'])
            # Get the coordinates and convert them to integer
            x_min = int(rectangle.attrib['left'])
            y_min = int(rectangle.attrib['top'])
            x_max = int(rectangle.attrib['right'])
            y_max = int(rectangle.attrib['bottom'])
            self.bboxes[roi_name] = np.array([x_min, y_min, x_max, y_max])
            # self.mask[y_min:y_max, x_min:x_max] = 1     ## redundant, should be deleted after testing

    def extract_from_circle(self):
        for circle in self.root.findall('.//TCircleShape'):
            roi_name = str(circle.attrib['name'])
            # Get the coordinates and convert them to integer
            x_min = int(circle.attrib['left'])
            y_min = int(circle.attrib['top'])
            x_max = int(circle.attrib['right'])
            y_max = int(circle.attrib['bottom'])
            self.bboxes[roi_name] = np.array([x_min, y_min, x_max, y_max])
            # self.mask[y_min:y_max, x_min:x_max] = 1

    def extract_from_polygon(self):
        points = {}
        for pie in self.root.findall('.//TPolygonShape'):
            roi_name = str(pie.attrib['name'])
            points[pie.attrib['name']] = {}
            points[pie.attrib['name']]["x"] = []
            points[pie.attrib['name']]["y"] = []
            for point in pie.findall("point"):
                x = int(point.attrib['x'])
                y = int(point.attrib['y'])
                points[pie.attrib['name']]["x"].append(x)
                points[pie.attrib['name']]["y"].append(y)
            x_min = min(points[pie.attrib['name']]["x"])
            y_min = min(points[pie.attrib['name']]["y"])
            x_max = max(points[pie.attrib['name']]["x"])
            y_max = max(points[pie.attrib['name']]["y"])
            self.bboxes[roi_name] = np.array([x_min, y_min, x_max, y_max])
            # self.mask[y_min:y_max, x_min:x_max] = 1

    def extract_from_pie(self):
        points = {}
        for pie in self.root.findall('.//TPieShape'):
            roi_name = str(pie.attrib['name'])
            points[pie.attrib['name']] = {}
            points[pie.attrib['name']]["x"] = []
            points[pie.attrib['name']]["y"] = []
            for point in pie.findall("point"):
                x = int(point.attrib['x'])
                y = int(point.attrib['y'])
                points[pie.attrib['name']]["x"].append(x)
                points[pie.attrib['name']]["y"].append(y)
            x_min = min(points[pie.attrib['name']]["x"])
            y_min = min(points[pie.attrib['name']]["y"])
            x_max = max(points[pie.attrib['name']]["x"])
            y_max = max(points[pie.attrib['name']]["y"])
            self.bboxes[roi_name] = np.array([x_min, y_min, x_max, y_max])
            # self.mask[y_min:y_max, x_min:x_max] = 1







def extract_chessboard_spots(rgb_image, mask_array, factor= 0):
    # Label connected components in the binary mask
    structure = np.ones((3, 3), dtype=int)  # Define connectivity (8-connectivity)
    labeled, num_features = label(mask_array, structure)
    
    # Extract the bounding slices for each component
    slices = find_objects(labeled)
    
    sub_images = []
    for slice_ in slices:
        # Expand each slice by 'expand_pixels' while ensuring we don't go out of image bounds
        start_y = max(slice_[0].start - factor, 0)
        stop_y = min(slice_[0].stop + factor, rgb_image.shape[0])
        start_x = max(slice_[1].start - factor, 0)
        stop_x = min(slice_[1].stop + factor, rgb_image.shape[1])
        
        # Create a new slice object for the expanded region
        expanded_slice = (slice(start_y, stop_y), slice(start_x, stop_x))
        # Extract the part of the RGB image corresponding to the current component
        sub_image = rgb_image[expanded_slice]
        
        # Append the extracted sub-image to the list
        sub_images.append(sub_image)

    return sub_images


def image_cutter(image, mask, x_coord, y_coord, orig_path, f=40):
    annotated_img = {}
    y_coord = [(y[0] - f, y[1]+f) for y in y_coord]
    x_coord = [(x[0] - f, x[1]+f) for x in x_coord]
    chunk_counter = 0
    for i, y in enumerate(y_coord):
        # print(y_coord)
        # print(i,y)
        for j, x in enumerate(x_coord):
            # print([y[0],y[1], x[0],x[1]])
        # Extract the sub-image using the current x and y coordinates
            sub_img = image[y[0]:y[1], x[0]:x[1], :]
            sub_mask = mask[y[0]:y[1], x[0]:x[1]]
            sub_bbox = np.array([[*find_bounding_box(sub_mask)]])
            # sub_mask = np.expand_dims(sub_mask, axis=0)
            base_name = os.path.basename(orig_path)
                # Remove the file extension from base_name
            image_name = os.path.splitext(base_name)[0]

            image_name = f"{image_name}_chunk{chunk_counter}"

            annotated_img[image_name] = [sub_img, sub_mask, sub_bbox]
            chunk_counter+=1
    return annotated_img


def mask_to_contour(output_dir, input_dict, save_orig_img = True):
    labels_dir = os.path.join(output_dir, 'labels')
    images_dir = os.path.join(output_dir, 'images')
    if os.path.exists(output_dir):
    # Remove the directory and all its contents
        pass
    else:
    # Create the new directory
        os.makedirs(output_dir)    
    # Create subdirectories 'labels' and 'images' within the main directory

        
        os.makedirs(labels_dir)
        os.makedirs(images_dir)

    for img in input_dict:
        mask = input_dict[img][1]
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        H, W = mask.shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # convert the contours to polygons
        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 200:
                polygon = []
                for point in cnt:
                    x, y = point[0]
                    polygon.append(x / W)
                    polygon.append(y / H)
                polygons.append(polygon)

        # print the polygons
        with open('{}.txt'.format(os.path.join(labels_dir, img)), 'w') as f:
            for polygon in polygons:
                for p_, p in enumerate(polygon):
                    if p_ == len(polygon) - 1:
                        f.write('{}\n'.format(p))
                    elif p_ == 0:
                        f.write('0 {} '.format(p))
                    else:
                        f.write('{} '.format(p))
        if save_orig_img is True:
            image = input_dict[img][0]
            cv2.imwrite('{}.jpg'.format(os.path.join(images_dir, img)), image)
            f.close()

def tray_mask_from_xml(xml_content):
    # Parse the XML content
    tree = ET.parse(xml_content)
    root = tree.getroot()
    
    # Retrieve the dimensions for the output array
    width = int(root.attrib['width'])
    height = int(root.attrib['height'])
    
    # Initialize the output array
    array = np.zeros((height, width), dtype=int)
    
    # Iterate through all rectangle shapes and fill the array
    for rectangle in root.find('TMultiShapes'):
        # Get the coordinates and convert them to integer
        left = int(rectangle.attrib['left'])
        top = int(rectangle.attrib['top'])
        right = int(rectangle.attrib['right'])
        bottom = int(rectangle.attrib['bottom'])
        
        # Fill the corresponding area in the array with ones
        array[top:bottom, left:right] = 1
    
    return array


def bbox_from_xml(xml_content):
    # Parse the XML content
    tree = ET.parse(xml_content)
    root = tree.getroot()
    bboxes = {}
    for rectangle in root.find('TMultiShapes'):
        # Get the coordinates and convert them to integer
        roi_name = str(rectangle.attrib['name'])
        x_min = int(rectangle.attrib['left'])
        y_min = int(rectangle.attrib['top'])
        x_max = int(rectangle.attrib['right'])
        y_max = int(rectangle.attrib['bottom'])
        bboxes[roi_name] = np.array([x_min, y_min, x_max, y_max])
    return bboxes


def create_mask_from_xml(xml_path, tensor=False):
    """
    Create a binary mask from polygons specified in an XML file.
    
    Args:
        xml_path (str): Path to the XML file.
        tensor (bool): If True, returns the mask as a PyTorch tensor. Otherwise, returns a numpy array.
        
    Returns:
        numpy.ndarray or torch.Tensor: The binary mask.
    """
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Create an empty binary mask
    width = int(root.attrib['width'])
    height = int(root.attrib['height'])
    mask = np.zeros((height, width), dtype=np.uint8)  # Black mask
    
    # Extract polygons and draw them on the mask
    for polygon in root.findall('.//TPolygonShape'):
        points = [(int(point.get('x')), int(point.get('y'))) for point in polygon.findall('.//point')]
        points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points], color=255)  # Fill the polygon with white
    
    # Convert to PyTorch tensor if requested
    if tensor:
        mask = torch.tensor(mask, dtype=torch.uint8)
    
    return mask

def remove_small_regions(binary_mask, min_size):
    """
    Removes small regions from a binary mask.
    
    :param binary_mask: 2D numpy array representing the binary mask.
    :param min_size: Integer, the minimum size of the regions to keep.
    :return: 2D numpy array with small regions removed.
    """
    labeled_mask, num_features = label(binary_mask)
    sizes = np.bincount(labeled_mask.ravel())
    
    mask_size = sizes >= min_size
    mask_size[0] = 0  # Ensure background (label 0) is not considered
    
    cleaned_mask = mask_size[labeled_mask]
    
    return cleaned_mask


def overlay_image_with_mask(image, mask, mask_color=(255, 0, 0), alpha=0.5, output_path=None, rgb_corr=False):
    """
    Overlay an RGB image with a binary mask, highlighting only the masked regions.
    
    Args:
        image (numpy.ndarray): Original RGB image.
        mask (numpy.ndarray): Binary mask array.
        mask_color (tuple): Color for the mask overlay as (B, G, R).
        alpha (float): Transparency factor for the mask. Range [0, 1].
        output_path (str): Path to save the overlayed image (optional).
        rgb_corr (bool): If True, corrects color space from BGR to RGB.
    
    Returns:
        numpy.ndarray: Image with mask overlay showing only the masked regions.
    """
    # Ensure the mask is boolean
    mask = mask.astype(bool)
    
    # Convert the mask color to a numpy array and scale it to [0, 1]
    mask_color = np.array(mask_color) / 255.0
    
    # Create a color version of the mask
    color_mask = np.zeros_like(image, dtype=float)
    color_mask[mask] = mask_color
    
    # Convert the original image to float for blending
    image_float = image.astype(float)
    
    # Blend the mask with the original image
    overlayed_image = image_float.copy()
    for c in range(3):  # Apply blending for each channel
        overlayed_image[:, :, c] = (1 - alpha) * image_float[:, :, c] + alpha * color_mask[:, :, c]
    
    # Convert back to uint8
    overlayed_image = (overlayed_image * 255).astype(np.uint8)
    
    if rgb_corr:
        overlayed_image = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)
    
    # Save the result if output path is provided
    if output_path:
        cv2.imwrite(output_path, overlayed_image)
        print(f"Overlayed image saved as {output_path}")
    
    return overlayed_image



def find_bounding_box(binary_mask):
    # Find the rows and columns where the mask has non-zero values
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    
    # Find indices of non-zero rows and columns
    y_min, y_max = np.where(rows)[0][0], np.where(rows)[0][-1]
    x_min, x_max = np.where(cols)[0][0], np.where(cols)[0][-1]
    # making it a bit wider
    y_min = max(y_min - 10, 0)
    y_max = min(y_max + 10, binary_mask.shape[0] - 10)
    x_min = max(x_min - 10, 0)
    x_max = min(x_max + 10, binary_mask.shape[1] - 10)
    
    # Return the coordinates of the bounding box: (x_min, y_min, x_max, y_max)
    return x_min, y_min, x_max, y_max


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a binary mask from an XML file.')
    parser.add_argument('xml_path', type=str, help='Path to the XML file containing the mask definition.')
    parser.add_argument('--tensor', action='store_true', help='Return the mask as a PyTorch tensor instead of a numpy array.')
    parser.add_argument('--save', type=str, default='', help='Path to save the output mask as a binary PNG image. If not specified, the mask is not saved.')

    args = parser.parse_args()

    # Create the mask
    mask = create_mask_from_xml(args.xml_path, tensor=args.tensor)
    
    # Save the mask if a save path is provided
    if args.save:
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()  # Convert to numpy array if it's a tensor
        cv2.imwrite(args.save, mask)
        print(f"Mask saved as {args.save}")
