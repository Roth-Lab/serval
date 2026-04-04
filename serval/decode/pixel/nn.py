"""
Module for k-nearest neighbor based pixel decoding.

This decoder uses sklearn's NearestNeighbors to assign each pixel an index
based on the nearest codebook vector in Euclidean space.
"""

from sklearn.neighbors import NearestNeighbors

import numpy as np

from serval.decode.pixel import PixelDecoder, PixelDecoderResult
from serval.decode.utils import normalize, reshape_data


class NearestNeigbourPixelDecoder(PixelDecoder):
    """Nearest neighbor pixel decoder.

    Uses k-NN to assign each pixel to the nearest barcode vector within a
    maximum distance and optional normalization.

    Attributes:
        max_dist (float): Maximum allowed distance for assignment; beyond this,
            pixels are labeled as background (-1).
        min_norm (float): Minimum vector norm to consider; pixels with norm
            below this are labeled as background.
        norm (bool): Whether to normalize pixel intensity vectors before decoding.
    """

    def __init__(self, codebook, max_dist=0.5176, min_norm=1, norm=True):
        """Initialize a NearestNeigbourPixelDecoder.

        Args:
            codebook (Codebook): Codebook defining barcode targets.
            max_dist (float): Distance threshold for valid assignment. Defaults to 0.5176.
            min_norm (float): Minimum vector norm for decoding. Defaults to 1.0.
            norm (bool): Whether to normalize vectors before nearest-neighbor search. Defaults to True.
        """
        super().__init__(codebook)

        self.converged = True

        self.max_dist = max_dist

        self.min_norm = min_norm

        self.norm = norm

        self.nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")

        if self.norm:
            self.nn.fit(normalize(self.codebook.barcode_matrix))

        else:
            self.nn.fit(self.codebook.barcode_matrix)

    # Interface
    def predict(self, imgs):
        """Decode an image stack into PixelDecoderResult using nearest neighbors.

        Args:
            imgs (ImageStack): Image stack with attributes `imgs`, `fov`, `z`.

        Returns:
            PixelDecoderResult: Contains `dist`, `idxs`, `imgs`, `norm`, and
                `info` with the preprocessed data `X`.
        """
        X = reshape_data(imgs.imgs)

        norm = np.linalg.norm(X, axis=1)

        if self.norm:
            norm[norm == 0] = 1

            X = X / norm[:, np.newaxis]

        dist, idxs = self.nn.kneighbors(X, return_distance=True)

        idxs[dist > self.max_dist] = -1

        idxs[norm < self.min_norm] = -1

        dist = dist.reshape(imgs.imgs[0].shape)

        norm = norm.reshape(imgs.imgs[0].shape)

        idxs = idxs.reshape(imgs.imgs[0].shape)

        X = np.moveaxis(X, 1, 0).reshape(imgs.imgs.shape)

        return PixelDecoderResult(
            self.codebook,
            dist,
            idxs,
            imgs,
            norm,
            info={"X": X},
        )
