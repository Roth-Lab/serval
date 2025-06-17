import random

import numpy as np
import scipy.optimize
import skimage

from serval.decode.pixel import PixelDecoder
from serval.image import ImageStack


class CosineOptimizedPixelDecoder(PixelDecoder):
    # TODO: Init scaling factors support?
    def __init__(
        self,
        codebook,
        decoder,
        fit_max_size=None,
        fit_max_area=12,
        fit_min_area=3,
        init_scaling_factors=None,
        penalty=0,
    ):
        super().__init__(codebook)

        self.decoder = decoder

        self.fit_max_size = fit_max_size

        self.fit_max_area = fit_max_area

        self.fit_min_area = fit_min_area

        if init_scaling_factors is None:
            init_scaling_factors = np.ones(codebook.num_bits)

        self.scaling_factors = init_scaling_factors

        self.penalty = penalty

        B = self.codebook.barcode_matrix

        self.norm_barcodes = B / np.linalg.norm(B, axis=1)[:, np.newaxis]

    # Interface
    @property
    def params(self):
        return self.scaling_factors.copy()

    @params.setter
    def params(self, x):
        self.scaling_factors = x

    def predict(self, imgs):
        return self.decoder.predict(self._get_scaled_img(imgs))

    def score(self, img):
        decoded = self.predict(img)

        X = [img[:, decoded.pixels == i] for i in range(self.codebook.num_targets)]

        return obj_func(self.scaling_factors, self.norm_barcodes, X)

    def get_update_params(self, local_params):
        def f(x, y, z):
            # return -1 * obj_func(x, y, z) + self.penalty * np.sum((x - 1) ** 2)
            x_bar = x / np.sum(x)
            return (
                -1 * obj_func(x, y, z)
                + self.penalty * np.sum(x_bar * np.log(x_bar))
                + self.penalty * np.sum((x - 1) ** 2)
            )

        def g(x, y, z):
            # return -1 * grad_obj_func(x, y, z) + 2 * self.penalty * (x - 1)
            x_nrm = np.sum(x)
            x_bar = x / x_nrm
            return (
                -1 * grad_obj_func(x, y, z)
                + self.penalty * (np.log(x_bar) + 1) * (np.sum(x_bar) - x_bar) / x_nrm
                + 2 * self.penalty * (x - 1)
            )

        X = [[] for _ in range(self.codebook.num_targets)]

        for x in local_params:
            for i in range(self.codebook.num_targets):
                X[i].extend(x[i])

        X = [np.stack(x, axis=1) if len(x) > 0 else [] for x in X]

        r = scipy.optimize.minimize(
            f,
            self.scaling_factors,
            args=(self.norm_barcodes, X),
            jac=g,
            bounds=[(0, None) for _ in range(self.codebook.num_bits)],
        )

        return r["x"]

    def get_local_update_params(self, imgs):
        # TODO: Add support for spot size filtering like the scaled decoder
        decoded = self.predict(imgs)

        coords = []

        for b in range(self.codebook.num_targets):
            barcode_regions = [
                x
                for x in skimage.measure.regionprops(
                    skimage.measure.label((decoded.idxs == b).astype(int))
                )
                if (self.fit_min_area <= x.area <= self.fit_max_area)
            ]

            for br in barcode_regions:
                coords.extend([(b, x, y) for x, y in br.coords])

        if (self.fit_max_size is not None) and (len(coords) > self.fit_max_size):
            coords = random.sample(coords, self.fit_max_size)

        result = [[] for _ in range(self.codebook.num_targets)]

        for b, x, y in coords:
            result[b].append(imgs.imgs[:, x, y])

        return result

    # Helper methods
    def _get_scaled_img(self, imgs):
        return ImageStack(
            imgs.imgs / self.scaling_factors[:, np.newaxis, np.newaxis],
            imgs.fov,
            imgs.z,
        )


def obj_func(c, B, X):
    if np.min(c) < 0:
        return -np.inf
    tot = 0
    for b, x in zip(B, X):
        if len(x) == 0:
            continue
        x = x / c[:, np.newaxis]
        x = x / np.linalg.norm(x, axis=0)[np.newaxis, :]
        x = np.sum(x, axis=1)
        tot += np.sum(b * x)
    return tot


def grad_obj_func(c, B, X):
    grad = np.zeros(c.shape)
    for b, x in zip(B, X):
        if len(x) == 0:
            continue
        x = x / c[:, np.newaxis]
        x = x / np.linalg.norm(x, axis=0)[np.newaxis, :]
        xb = x * b[:, np.newaxis]
        xx = x * x
        xx_xb = xx * np.sum(xb, axis=0)[np.newaxis, :]
        grad += np.sum((xx_xb - xb), axis=1)
    grad = grad / c
    return grad
