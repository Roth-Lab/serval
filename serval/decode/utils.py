"""
Utility functions for image cropping, normalization, reshaping, and histogram-based scaling.
"""
import numpy as np
import skimage.util


def crop_image(img, crop_size=0):
    """Crop each frame in an image stack by a specified number of pixels.
    
    Args:
        img (np.ndarray): Image stack array with shape (frames, height, width).
        crop_size (int): Number of pixels to remove from each edge. Defaults to 0.
    
    Returns:
        np.ndarray: Cropped image stack with the same number of frames.
    """
    if crop_size > 0:
        img_new = []

        for i in range(img.shape[0]):
            img_new.append(skimage.util.crop(img[i], crop_size))

        img = np.array(img_new)

    return img


def normalize(x):
    """Normalize rows of a 2D array to unit length.
    
    Args:
        x (np.ndarray): Input array of shape (n_vectors, vector_length).
    
    Returns:
        np.ndarray: Row-normalized array where each row has Euclidean norm 1.
    """
    norm = np.linalg.norm(x, axis=1)

    norm[norm == 0] = 1

    return x / norm[:, np.newaxis]


def reshape_data(imgs):
    """Reshape an image stack into a 2D array suitable for vectorized operations.
    
    Args:
        imgs (np.ndarray): Image stack of shape (n_frames, height, width).
    
    Returns:
        np.ndarray: 2D array of shape (height*width, n_frames).
    """
    return imgs.reshape((imgs.shape[0], np.prod(imgs.shape[1:]))).T


def get_init_scaling_factors(hists, q=0.9):
    """Given pixel intensity histograms computed across each bit, compute initial scaling factors
    
    Basically computes the quantile.    
    
    Args:
        hists (np.ndarray): Histogram counts array of shape (n_bits, n_bins).
        q (float): Quantile to use for scaling factor calculation (0 < q < 1). Defaults to 0.9.
    
    Returns:
        np.ndarray: Array of scaling factors of length n_bits.
    """
    num_bits = hists.shape[0]

    scaling_factors = np.zeros(num_bits)

    for b in range(num_bits):
        cum_dist = hists[b].cumsum()

        prob_dist = cum_dist / cum_dist[-1]

        # Note: This should probably be +1 but +2 matches MERLIn
        scaling_factors[b] = np.argmin(np.abs(prob_dist - q)) + 2

    return scaling_factors


def get_imgs_hist(imgs, transforms):
    """Compute pixel intensity histograms for each frame/imaging round of an image stack after applying transforms.
    
    Args:
        imgs (ImageStack): An ImageStack instance with attributes `imgs` and `num_frames`.
        transforms (list): List of objects with a `transform` method accepting and returning an ImageStack.
    
    Returns:
        np.ndarray: Array of shape (n_frames, n_bins) containing histograms.
    """
    for t in transforms:
        imgs = t.transform(imgs)

    bins = np.arange(np.iinfo(np.uint16).max)

    hist = []

    for i in range(imgs.num_frames):
        hist.append(np.histogram(imgs.imgs[i], bins=bins)[0])

    return np.stack(hist, axis=0)
