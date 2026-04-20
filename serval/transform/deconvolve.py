"""
Module for deconvolution-based image transforms using Lucy-Richardson algorithm.

This module provides:
- DeconvoleImageTransform: ImageTransform performing iterative deconvolution.
- lucy_richardson_deconvolve: Core deconvolution function.
- gaussian_2d: Utility to generate a 2D Gaussian kernel.
"""
import cv2
import numpy as np
import scipy.ndimage

from serval.transform import ImageTransform


class DeconvoleImageTransform(ImageTransform):
    """ImageTransform that applies Lucy-Richardson deconvolution to each frame.
    
    Attributes:
        filter_size (int, optional): Size of the deconvolution kernel. Defaults based on sigma.
        num_iters (int): Number of Richardson-Lucy iterations. Defaults to 20.
        sigma (float): Standard deviation for Gaussian PSF. Defaults to 1.0.
    """
    def __init__(self, filter_size=None, num_iters=20, sigma=1):
        """Initialize DeconvoleImageTransform.
        
        Args:
            filter_size (int, optional): Kernel size; if None, computed from sigma.
            num_iters (int): Number of deconvolution iterations. Defaults to 20.
            sigma (float): Standard deviation for PSF Gaussian. Defaults to 1.0.
        """
        if filter_size is None:
            filter_size = int(2 * np.ceil(2 * sigma) + 1)

        self.filter_size = filter_size

        self.num_iters = num_iters

        self.sigma = sigma

    def _transform(self, img, bit, fov, z):
        """Apply Lucy-Richardson deconvolution to a single frame.
        
        Args:
            img (np.ndarray): 2D image array, will be cast to float internally.
            bit (int): Bit index (unused).
            fov (int): Field-of-view index (unused).
            z (int): Z-slice index (unused).
        
        Returns:
            np.ndarray: Deconvolved image after num_iters iterations.
        """
        return lucy_richardson_deconvolve(
            img.astype(float),
            num_iters=self.num_iters,
            sigma=self.sigma,
            window_size=self.filter_size,
        )


def lucy_richardson_deconvolve(img, num_iters=20, sigma=2.0, window_size=None):
    """Perform Lucy-Richardson deconvolution.
    
    Ported from https://github.com/emanuega/MERlin/blob/master/merlin/util/deconvolve.py which was ported from Matlab deconvlucy.
    
    Args:
        img (np.ndarray): 2D float image to deconvolve.
        num_iters (int): Number of iterations. Defaults to 20.
        sigma (float): Standard deviation for Gaussian PSF. Defaults to 2.0.
        window_size (int, optional): Kernel size; if None, computed from sigma.
    
    Returns:
        np.ndarray: Deconvolved image after specified iterations.
    """
    if window_size is None:
        window_size = int(2 * np.ceil(2 * sigma) + 1)

    eps = np.finfo(float).eps
    Y = np.copy(img)
    J1 = np.copy(img)
    J2 = np.copy(img)
    wI = np.copy(img)
    imR = np.copy(img)
    reblurred = np.copy(img)
    tmpMat1 = np.zeros(img.shape, dtype=float)
    tmpMat2 = np.zeros(img.shape, dtype=float)
    T1 = np.zeros(img.shape, dtype=float)
    T2 = np.zeros(img.shape, dtype=float)
    l = 0

    if window_size % 2 != 1:
        gaussian_filter = gaussian_2d(shape=(window_size, window_size), sigma=sigma)

    for i in range(num_iters):
        if i > 1:
            cv2.multiply(T1, T2, tmpMat1)
            cv2.multiply(T2, T2, tmpMat2)
            l = np.sum(tmpMat1) / (np.sum(tmpMat2) + eps)
            l = max(min(l, 1), 0)
        cv2.subtract(J1, J2, Y)
        cv2.addWeighted(J1, 1, Y, l, 0, Y)
        np.clip(Y, 0, None, Y)
        if window_size % 2 == 1:
            cv2.GaussianBlur(
                Y,
                (window_size, window_size),
                sigma,
                reblurred,
                borderType=cv2.BORDER_REPLICATE,
            )
        else:
            reblurred = scipy.ndimage.convolve(Y, gaussian_filter, mode="constant")
        np.clip(reblurred, eps, None, reblurred)
        cv2.divide(wI, reblurred, imR)
        imR += eps
        if window_size % 2 == 1:
            cv2.GaussianBlur(
                imR,
                (window_size, window_size),
                sigma,
                imR,
                borderType=cv2.BORDER_REPLICATE,
            )
        else:
            imR = scipy.ndimage.convolve(imR, gaussian_filter, mode="constant")
            imR[imR > 2**16] = 0
        np.copyto(J2, J1)
        np.multiply(Y, imR, out=J1)
        np.copyto(T2, T1)
        np.subtract(J1, Y, out=T1)
    return J1


def gaussian_2d(shape=(3, 3), sigma=0.5):
    """Generate a 2D Gaussian kernel.
    
    Ported from https://github.com/emanuega/MERlin/blob/master/merlin/util/matlab.py
    
    Args:
        shape (tuple of int): Kernel shape (height, width).
        sigma (float): Standard deviation of the Gaussian. Defaults to 0.5.
    
    Returns:
        np.ndarray: Normalized 2D Gaussian kernel array.
    """
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
