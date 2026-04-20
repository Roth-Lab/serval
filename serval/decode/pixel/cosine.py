"""
Module for cosine-optimized pixel decoding.

Implements a PixelDecoder that iteratively optimizes per-bit scaling factors
to maximize cosine similarity between pixel intensity vectors and codebook barcodes.
"""

import random

import numpy as np
import scipy.optimize
import skimage

from serval.decode.pixel import PixelDecoder
from serval.image import ImageStack


def obj_func(c, B, X):
    """Objective function: sum of cosine similarities across bits.

    Args:
        c (np.ndarray): Scaling factors array.
        B (np.ndarray): Normalized barcode matrix (targets × bits).
        X (list of np.ndarray): Pixel intensity matrices for each target.

    Returns:
        float: Total cosine similarity score.
    """
    if np.min(c) < 0:
        return -np.inf
    tot = 0
    for b, x in zip(B, X):
        n = len(x)
        if n == 0:
            continue
        x = x / c[:, np.newaxis]
        x = x / np.linalg.norm(x, axis=0)[np.newaxis, :]
        x = np.sum(x, axis=1)
        tot += (1 / n) * np.sum(b * x)
    return tot


def grad_obj_func(c, B, X):
    """Gradient of the cosine similarity objective w.r.t. scaling factors.

    Args:
        c (np.ndarray): Scaling factors array.
        B (np.ndarray): Normalized barcode matrix (targets × bits).
        X (list of np.ndarray): Pixel intensity matrices for each target.

    Returns:
        np.ndarray: Gradient vector of same shape as c.
    """
    grad = np.zeros(c.shape)
    for b, x in zip(B, X):
        n = len(x)
        if n == 0:
            continue
        x = x / c[:, np.newaxis]
        x = x / np.linalg.norm(x, axis=0)[np.newaxis, :]
        xb = x * b[:, np.newaxis]
        xx = x * x
        xx_xb = xx * np.sum(xb, axis=0)[np.newaxis, :]
        grad += (1 / n) * np.sum((xx_xb - xb), axis=1)
    grad = grad / c
    return grad


class CosineOptimizedPixelDecoder(PixelDecoder):
    """Cosine-optimized pixel decoder.

    Wraps another PixelDecoder and optimizes scaling factors via cosine similarity
    objective with entropy and L2 penalties.

    Attributes:
        decoder (PixelDecoder): Underlying decoder used after scaling.
        scaling_factors (np.ndarray): Current per-bit scaling factors.
        penalty_entropy (float): Regularization strength on scaling factors.
        fit_max_size (int or None): Max number of pixel locations to sample when fitting.
        fit_max_area (int): Max spot area (in pixels) to include when sampling.
        fit_min_area (int): Min spot area (in pixels) to include when sampling.
        norm_barcodes (np.ndarray): Array of shape (n_barcodes, n_bits) where each row of the
            barcode matrix B has been L2‐normalized.
    """

    def __init__(
        self,
        codebook,
        decoder,
        fit_max_size=None,
        fit_max_area=12,
        fit_min_area=3,
        init_scaling_factors=None,
        penalty_entropy=0,
        penalty_l2=0,
        pre_scaling_factors=None,
    ):
        """Initialize a CosineOptimizedPixelDecoder.

        Args:
            codebook (Codebook): Codebook instance providing target names.
            decoder (PixelDecoder): Underlying decoder to refine.
            fit_max_size (int, optional): Max sample size for fitting. Defaults to None.
            fit_max_area (int): Max spot area for sampling. Defaults to 12.
            fit_min_area (int): Min spot area for sampling. Defaults to 3.
            init_scaling_factors (np.ndarray, optional): Initial scaling factors array. Defaults to ones.
            penalty_entropy (float): Regularization strength. Defaults to 0.0.
        """
        super().__init__(codebook)

        self.decoder = decoder

        self.fit_max_size = fit_max_size

        self.fit_max_area = fit_max_area

        self.fit_min_area = fit_min_area

        if init_scaling_factors is None:
            init_scaling_factors = np.ones(codebook.num_bits)

        self.scaling_factors = init_scaling_factors

        self.penalty_entropy = penalty_entropy

        self.penalty_l2 = penalty_l2

        self.pre_scaling_factors = pre_scaling_factors

        B = self.codebook.barcode_matrix

        self.norm_barcodes = B / np.linalg.norm(B, axis=1)[:, np.newaxis]

    # Interface
    @property
    def params(self):
        """np.ndarray: Copy of current per-bit scaling factors."""
        return self.scaling_factors.copy()

    @params.setter
    def params(self, x):
        """Set per-bit scaling factors.

        Args:
            x (np.ndarray): New scaling factors of length codebook.num_bits.
        """
        print(self.scaling_factors)

        print(x)

        diff = np.linalg.norm(x - self.scaling_factors)

        print(diff)

        if diff == 0:
            self.converged = True

        self.scaling_factors = x

    def predict(self, imgs):
        """Decode scaled image stack using the underlying decoder.

        Args:
            imgs (ImageStack): Original image stack.

        Returns:
            PixelDecoderResult: Result from the underlying decoder.
        """
        imgs = self._get_pre_scaled_imgs(imgs)

        return self.decoder.predict(self._get_scaled_img(imgs))

    def score(self, img):
        """Compute objective function value for a single image frame.

        Args:
            img (ImageStack): Image stack or frame to score.

        Returns:
            float: Cosine objective value minus regularization.
        """
        decoded = self.predict(img)

        X = [img[:, decoded.pixels == i] for i in range(self.codebook.num_targets)]

        return obj_func(self.scaling_factors, self.norm_barcodes, X)

    def get_update_params(self, local_params):
        """Optimize scaling factors based on collected local parameters.

        Args:
            local_params (list): Per-frame scaling updates.

        Returns:
            np.ndarray: Updated scaling factors after optimization.
        """

        def f(x, y, z):
            x_bar = x / np.sum(x)
            return (
                -1 * obj_func(x, y, z)
                + self.penalty_entropy * N * np.sum(x_bar * np.log(x_bar))
                + self.penalty_l2 * N * np.sum((x - 1) ** 2)
            )

        def g(x, y, z):
            x_nrm = np.sum(x)
            x_bar = x / x_nrm
            return (
                -1 * grad_obj_func(x, y, z)
                + self.penalty_entropy * N * (np.log(x_bar) + 1) * (np.sum(x_bar) - x_bar) / x_nrm
                + 2 * self.penalty_l2 * N * (x - 1)
            )

        X = [[] for _ in range(self.codebook.num_targets)]

        for x in local_params:
            for i in range(self.codebook.num_targets):
                X[i].extend(x[i])

        X = [np.stack(x, axis=1) if len(x) > 0 else [] for x in X]

        N = sum([len(x) for x in X])

        r = scipy.optimize.minimize(
            f,
            self.scaling_factors,
            args=(self.norm_barcodes, X),
            jac=g,
            bounds=[(1e-6, None) for _ in range(self.codebook.num_bits)],
        )

        return r["x"]

    def get_local_update_params(self, imgs):
        """Collect pixel traces for spots meeting area criteria.

        Args:
            imgs (ImageStack): Scaled image stack to analyze.

        Returns:
            list of np.ndarray: Pixel intensity traces grouped by barcode.
        """
        decoded = self.predict(imgs)

        coords = []

        for b in range(self.codebook.num_targets):
            barcode_regions = [
                x
                for x in skimage.measure.regionprops(skimage.measure.label((decoded.idxs == b).astype(int)))
                if (self.fit_min_area <= x.area <= self.fit_max_area)
            ]

            for br in barcode_regions:
                coords.extend([(b, x, y) for x, y in br.coords])

        if (self.fit_max_size is not None) and (len(coords) > self.fit_max_size):
            coords = random.sample(coords, self.fit_max_size)

        result = [[] for _ in range(self.codebook.num_targets)]

        imgs = self._get_pre_scaled_imgs(imgs)

        for b, x, y in coords:
            result[b].append(imgs.imgs[:, x, y])

        return result

    # Helper methods
    def _get_pre_scaled_imgs(self, imgs):
        if self.pre_scaling_factors is not None:
            imgs = ImageStack(
                imgs.imgs / self.pre_scaling_factors[:, np.newaxis, np.newaxis],
                imgs.fov,
                imgs.z,
            )

        return imgs

    def _get_scaled_img(self, imgs):
        """Apply current scaling factors to image stack.

        Args:
            imgs (ImageStack): Original image stack.

        Returns:
            ImageStack: Scaled image stack for decoding.
        """
        return ImageStack(
            imgs.imgs / self.scaling_factors[:, np.newaxis, np.newaxis],
            imgs.fov,
            imgs.z,
        )
