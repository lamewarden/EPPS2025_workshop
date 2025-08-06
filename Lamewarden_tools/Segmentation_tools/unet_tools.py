import torch.nn as nn
# from PIL import Image
import torch
import numpy as np
from math import ceil
from torch.nn.functional import softmax
from skimage import img_as_float32
from skimage.exposure import rescale_intensity

def pad(image, width: int, mode='reflect', constant_values=0):
    # only pad the first two dimensions
    pad_width = [(width, width), (width, width)]
    if len(image.shape) == 3:
        # don't pad channels
        pad_width.append((0, 0))
    if mode == 'reflect':
        return np.pad(image, pad_width, mode)
    return np.pad(image, pad_width, mode=mode,
                         constant_values=constant_values)


def get_tiles(image, in_tile_shape, out_tile_shape):
    width_diff = in_tile_shape[1] - out_tile_shape[1]
    pad_width = width_diff // 2
    padded_photo = pad(image, pad_width)

    horizontal_count = ceil(image.shape[1] / out_tile_shape[1])
    vertical_count = ceil(image.shape[0] / out_tile_shape[0])

    # first split the image based on the tiles that fit
    x_coords = [h*out_tile_shape[1] for h in range(horizontal_count-1)]
    y_coords = [v*out_tile_shape[0] for v in range(vertical_count-1)]

    # The last row and column of tiles might not fit
    # (Might go outside the image)
    # so get the tile positiion by subtracting tile size from the
    # edge of the image.
    right_x = padded_photo.shape[1] - in_tile_shape[1]
    bottom_y = padded_photo.shape[0] - in_tile_shape[0]

    y_coords.append(bottom_y)
    x_coords.append(right_x)

    # because its a rectangle get all combinations of x and y
    tile_coords = [(x, y) for x in x_coords for y in y_coords]
    tiles = tiles_from_coords(padded_photo, tile_coords, in_tile_shape)
    return tiles, tile_coords

def tiles_from_coords(image, coords, tile_shape):
    tiles = []
    for x, y in coords:
        tile = image[y:y+tile_shape[0],
                     x:x+tile_shape[1]]
        tiles.append(tile)
    return tiles

def get_valid_patch_sizes():
    return list((572 - (16*i) for i in range(31)))

def reconstruct_from_tiles(tiles, coords, output_shape):
    image = np.zeros(output_shape)
    for tile, (x, y) in zip(tiles, coords):
        image[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
    return image

def load_model(model_path):
    model = UNetGNRes()
    if torch.cuda.is_available():
        try:
            model.load_state_dict(torch.load(model_path))
            model = torch.nn.DataParallel(model)
        except:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(model_path))
        model.cuda()
    else:
        # if you are running on a CPU-only machine, please use torch.load with 
        # map_location=torch.device('cpu') to map your storages to the CPU.
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model = torch.nn.DataParallel(model)
        except:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

def pad_to_min(im, min_w, min_h):
    h, w, _ = im.shape
    h_pad = 0
    w_pad = 0
    if h < min_h:
        h_pad = min_h - h
    if w < min_w:
        w_pad = min_w - w
    h_pad_before = h_pad // 2
    h_pad_after = h_pad - h_pad_before
    w_pad_before = w_pad // 2
    w_pad_after = w_pad - w_pad_before
    pad_settings = ((h_pad_before, h_pad_after), (w_pad_before, w_pad_after), (0, 0))
    if h_pad or w_pad:
        im = np.pad(im, pad_settings, mode='reflect')
    return im, pad_settings

def ensemble_segment(model_paths, image, bs, in_w, out_w,
                     threshold=0.5):
    """ Average predictions from each model specified in model_paths """
    pred_sum = None
    pred_count = 0
    image, pad_settings = pad_to_min(image, min_w=in_w, min_h=in_w)
    # then add predictions from the previous models to form an ensemble
    for model_path in model_paths:
        cnn = load_model(model_path)
        preds = unet_segment(cnn, image,
                             bs, in_w, out_w, threshold=None)
        if pred_sum is not None:
            pred_sum += preds
        else:
            pred_sum = preds
        pred_count += 1
        # get flipped version too (test time augmentation)
        flipped_im = np.fliplr(image)
        flipped_pred = unet_segment(cnn, flipped_im, bs, in_w,
                                    out_w, threshold=None)
        pred_sum += np.fliplr(flipped_pred)
        pred_count += 1
    pred_sum = crop_from_pad_settings(pred_sum, pad_settings)
    foreground_probs = pred_sum / pred_count
    predicted = foreground_probs > threshold
    predicted = predicted.astype(int)
    return predicted

def crop_from_pad_settings(image, pad_settings):
    """ Crop image back to what it was before padding using
        the pad_settings. See pad_to_min for how 
        pad_settings are generated and used """
    h_pad_before, h_pad_after = pad_settings[0]
    w_pad_before, w_pad_after = pad_settings[1]
    h_start = h_pad_before
    h_stop = image.shape[0] - h_pad_after
    w_start = w_pad_before
    w_stop = image.shape[1] - w_pad_after
    return image[h_start:h_stop, w_start:w_stop]

def normalize_tile(tile):
    if np.min(tile) < np.max(tile):
        tile = rescale_intensity(tile, out_range=(0, 1))
    assert np.min(tile) >= 0, f"tile min {np.min(tile)}"
    assert np.max(tile) <= 1, f"tile max {np.max(tile)}"
    return tile

def unet_segment(cnn, image, bs, in_w, out_w, threshold=0.5):
    """
    Threshold set to None means probabilities returned without thresholding.
    """
    assert image.shape[0] >= in_w, str(image.shape[0])
    assert image.shape[1] >= in_w, str(image.shape[1])

    tiles, coords = get_tiles(image,
                                       in_tile_shape=(in_w, in_w, 3),
                                       out_tile_shape=(out_w, out_w))
    tile_idx = 0
    batches = []
    while tile_idx < len(tiles):
        tiles_to_process = []
        for _ in range(bs):
            if tile_idx < len(tiles):
                tile = tiles[tile_idx]
                tile = img_as_float32(tile)
                tile = normalize_tile(tile)
                tile = np.moveaxis(tile, -1, 0)
                tile_idx += 1
                tiles_to_process.append(tile)
        tiles_for_gpu = torch.from_numpy(np.array(tiles_to_process))
        if torch.cuda.is_available():
            tiles_for_gpu.cuda()
        tiles_for_gpu = tiles_for_gpu.float()
        batches.append(tiles_for_gpu)

    output_tiles = []
    for gpu_tiles in batches:
        outputs = cnn(gpu_tiles)
        softmaxed = softmax(outputs, 1)
        foreground_probs = softmaxed[:, 1, :]  # just the foreground probability.
        if threshold is not None:
            predicted = foreground_probs > threshold
            predicted = predicted.view(-1).int()
        else:
            predicted = foreground_probs

        pred_np = predicted.data.cpu().numpy()
        out_tiles = pred_np.reshape((len(gpu_tiles), out_w, out_w))
        for out_tile in out_tiles:
            output_tiles.append(out_tile)

    assert len(output_tiles) == len(coords), (
        f'{len(output_tiles)} {len(coords)}')

    reconstructed = reconstruct_from_tiles(output_tiles, coords,
                                                    image.shape[:-1])
    return reconstructed


def crop_tensor(tensor, target):
    """ Crop tensor to target size """
    _, _, tensor_height, tensor_width = tensor.size()
    _, _, crop_height, crop_width = target.size()
    left = (tensor_width - crop_height) // 2
    top = (tensor_height - crop_width) // 2
    right = left + crop_width
    bottom = top + crop_height
    cropped_tensor = tensor[:, :, top: bottom, left: right]
    return cropped_tensor

class DownBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels*2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels*2,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels*2)
        )
        self.conv1x1 = nn.Sequential(
            # down sample channels again.
            nn.Conv2d(in_channels*2, in_channels,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        out1 = self.pool(x)
        out2 = self.conv1(out1)
        out3 = self.conv2(out2)
        out4 = self.conv1x1(out3)
        return out4 + out1

class UpBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels,
                               kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, in_channels)
        )

    def forward(self, x, down_out):
        out = self.conv1(x)
        cropped = crop_tensor(down_out, out)
        out = cropped + out # residual
        out = self.conv2(out)
        out = self.conv3(out)
        return out
    
class UNetGNRes(nn.Module):
    def __init__(self, im_channels=3):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(im_channels, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, 64),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.GroupNorm(32, 64)
            # now at 568 x 568, 64 channels
        )
        self.down1 = DownBlock(64)
        self.down2 = DownBlock(64)
        self.down3 = DownBlock(64)
        self.down4 = DownBlock(64)
        self.up1 = UpBlock(64)
        self.up2 = UpBlock(64)
        self.up3 = UpBlock(64)
        self.up4 = UpBlock(64)
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.GroupNorm(2, 2)
        )

    def forward(self, x):
        out1 = self.conv_in(x)
        out2 = self.down1(out1)
        out3 = self.down2(out2)
        out4 = self.down3(out3)
        out5 = self.down4(out4)
        out = self.up1(out5, out4)
        out = self.up2(out, out3)
        out = self.up3(out, out2)
        out = self.up4(out, out1)
        out = self.conv_out(out)
        return out