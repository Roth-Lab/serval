import cv2
import numpy as np

from serval.transform import ImageTransform


class HighPassImageTransform(ImageTransform):
    def __init__(self, sigma=1):
        self.sigma = sigma

        self.filter_size = int(2 * np.ceil(2 * sigma) + 1)

    def _transform(self, img, bit, fov, z):
        return high_pass_filter(img, sigma=self.sigma, window_size=self.filter_size)


def high_pass_filter(img, sigma, window_size):
    low_pass = cv2.GaussianBlur(
        img, (window_size, window_size), sigma, borderType=cv2.BORDER_REPLICATE
    )

    high_pass = img - low_pass

    high_pass[low_pass > img] = 0

    return high_pass
