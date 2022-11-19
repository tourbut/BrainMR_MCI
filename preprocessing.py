import numpy as np
from scipy import ndimage


def crop_image(image):
    # Create a mask with the background pixels
    mask = image == 0
    # Find the brain area
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)
    # Remove the background
    croped_image = image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1],top_left[2]:bottom_right[2]]
    
    return croped_image

def add_pad(image, pad_top = 10,pad_left = 10,pad_depth=10):
    
    height, width, depth = image.shape
    new_height = height + pad_top*2
    new_width = width + pad_left*2
    new_depth = depth + pad_depth*2
    final_image = np.zeros((new_height, new_width, new_depth))
    final_image[pad_top:pad_top + height, pad_left:pad_left + width,pad_depth:depth + pad_depth] = image
    
    return final_image

def z_score(image):
    """
    z-score nomalization
    """
    mask_image = image>image.mean()
    logical_mask = mask_image>0.
    mean = image[logical_mask].mean()
    std = image[logical_mask].std()
    return (image-mean)/std

def minmax(image,min_percentile = 5,max_percentile=95):
    """
    min-max nomalization
    """
    min_value = np.percentile(image,min_percentile)
    max_value = np.percentile(image,max_percentile)
    output =  (image - min_value) / (max_value - min_value)
    return output

def normalize(volume, min = -500, max = 700):
    """Normalize the volume"""
    
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    
    return window_image

def resize(img,desired_width=128,desired_height=128,desired_depth=128):

    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth 
    width_factor = 1 / width 
    height_factor = 1 / height
    # Rotate
    #img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)

    return img
