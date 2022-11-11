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
    croped_image = image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1],:]
    
    return croped_image

def add_pad(image, new_height=256, new_width=256):
    
    height, width, depth = image.shape
    final_image = np.zeros((new_height, new_width, depth))
    pad_left = int((new_width - width) // 2)
    pad_top = int((new_height - height) // 2)
    # Replace the pixels with the image's pixels
    final_image[pad_top:pad_top + height, pad_left:pad_left + width,:] = image
    
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

def normalize(volume, min = -500, max = 700):
    """Normalize the volume"""
    
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

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
