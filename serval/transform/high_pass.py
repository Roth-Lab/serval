"""
Module for high-pass filtering of image frames.

This module provides:
- HighPassImageTransform: ImageTransform that applies a high-pass filter by subtracting a Gaussian blur.
"""
import cv2
import numpy as np

from serval.transform import ImageTransform


class HighPassImageTransform(ImageTransform):
    """ImageTransform that applies a high-pass filter to each frame.
    
    Attributes:
        sigma (float): Gaussian blur standard deviation.
        filter_size (int): Size of the Gaussian kernel computed from sigma.
    """
    def __init__(self, sigma=1):
        """Initialize HighPassImageTransform.
        
        Args:
            sigma (float): Gaussian blur standard deviation. Defaults to 1.0.
        """
        self.sigma = sigma

        self.filter_size = int(2 * np.ceil(2 * sigma) + 1)

    def _transform(self, img, bit, fov, z):
        """Apply high-pass filter to a single frame.
        
        Args:
            img (np.ndarray): 2D image array.
            bit (int): Bit index (unused).
            fov (int): Field-of-view index (unused).
            z (int): Z-slice index (unused).
        
        Returns:
            np.ndarray: High-pass filtered image.
        """
        return high_pass_filter(img, sigma=self.sigma, window_size=self.filter_size)


def high_pass_filter(img, sigma, window_size):
    """Compute high-pass filtered image by subtracting a low-pass (Gaussian) version.
    
    Args:
        img (np.ndarray): 2D input image.
        sigma (float): Gaussian blur standard deviation.
        window_size (int): Size of the Gaussian kernel (must be odd).
    
    Returns:
        np.ndarray: High-pass filtered image, negative values clipped to zero.
    """
    low_pass = cv2.GaussianBlur(
        img, (window_size, window_size), sigma, borderType=cv2.BORDER_REPLICATE
    )

    high_pass = img - low_pass

    high_pass[low_pass > img] = 0

    return high_pass
