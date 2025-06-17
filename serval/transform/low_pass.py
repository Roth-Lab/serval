import cv2
import numpy as np

from serval.transform import ImageTransform


class LowPassImageTransform(ImageTransform):
    def __init__(self, sigma=1):
        self.sigma = sigma

        self.filter_size = int(2 * np.ceil(2 * sigma) + 1)

    def _transform(self, img, bit, fov, z):
        return low_pass_filter(img, sigma=self.sigma, window_size=self.filter_size)


def low_pass_filter(img, sigma, window_size):
    return cv2.GaussianBlur(img, (window_size, window_size), sigma)
