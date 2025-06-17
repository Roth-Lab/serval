from sklearn.neighbors import NearestNeighbors

import numpy as np

from serval.decode.pixel import PixelDecoder, PixelDecoderResult
from serval.decode.utils import normalize, reshape_data


class NearestNeigbourPixelDecoder(PixelDecoder):
    def __init__(self, codebook, max_dist=0.5176, min_norm=1, norm=True):
        super().__init__(codebook)

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
