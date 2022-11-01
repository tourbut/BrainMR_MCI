import numpy as np

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