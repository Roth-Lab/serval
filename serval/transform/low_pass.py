"""
Module for low-pass filtering of image frames.

This module provides:
- LowPassImageTransform: ImageTransform that applies a Gaussian blur.
"""

import cv2
import numpy as np

from serval.transform import ImageTransform


class LowPassImageTransform(ImageTransform):
    """ImageTransform that applies a low-pass (Gaussian) filter to each frame.

    Attributes:
        sigma (float): Standard deviation for Gaussian blur.
        filter_size (int): Size of the Gaussian kernel calculated from sigma.
    """

    def __init__(self, sigma=1):
        """Initialize LowPassImageTransform.

        Args:
            sigma (float): Gaussian blur standard deviation. Defaults to 1.0.
        """
        self.sigma = sigma

        self.filter_size = int(2 * np.ceil(2 * sigma) + 1)

    def _transform(self, img, bit, fov, z):
        """Apply low-pass filter to a single frame.

        Args:
            img (np.ndarray): 2D image array.
            bit (int): Bit index (unused).
            fov (int): Field-of-view index (unused).
            z (int): Z-slice index (unused).

        Returns:
            np.ndarray: Low-pass filtered image.
        """
        return low_pass_filter(img, sigma=self.sigma, window_size=self.filter_size)


def low_pass_filter(img, sigma, window_size):
    """Apply Gaussian low-pass filter to the image.

    Args:
        img (np.ndarray): 2D input image.
        sigma (float): Gaussian blur standard deviation.
        window_size (int): Size of the Gaussian kernel (must be odd).

    Returns:
        np.ndarray: Blurred image.
    """
    return cv2.GaussianBlur(img, (window_size, window_size), sigma)
