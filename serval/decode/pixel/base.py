"""
Utilities for decoding pixel-based MERlin results into spot DataFrames, and
base interface for pixel decoding algorithms.

This module provides:
- PixelDecoderResult: encapsulates decoding outputs and lazily generates spot tables.
- PixelDecoder: abstract interface defining `predict`, `fit`, and `score` methods.
"""

import numpy as np
import pandas as pd
import skimage


class PixelDecoderResult(object):
    """Encapsulates the result of a pixel decoding run.

    Spot extraction is performed lazily when accessing the `spots` property.

    Attributes:
        codebook (Codebook): The Codebook used for decoding targets.
        dist (np.ndarray): 2D array (height × width) of distances between each pixel’s
            intensity vector and its assigned barcode vector.
        idxs (np.ndarray): 2D integer array (height × width) assigning each pixel
            to a barcode index.
        imgs (np.ndarray): 3D array (bits × height × width) of original image frames.
        norm (np.ndarray): 2D array (height × width) of normalized intensities for spot scoring.
        info (dict, optional): Additional metadata about the decoding run.
    """

    def __init__(self, codebook, dist, idxs, imgs, norm, info=None):
        """Initialize a PixelDecoderResult.

        Args:
            codebook (Codebook): Codebook instance providing target names.
            dist (np.ndarray): Array of shape (H, W) with distance values per pixel.
            idxs (np.ndarray): Array of shape (H, W) with decoded barcode indices.
            imgs (np.ndarray): Array of shape (B, H, W) with raw image bit planes.
            norm (np.ndarray): Array of shape (H, W) of intensity values for spot metrics.
            info (dict, optional): Extra information about the decoding run. Defaults to None.
        """
        self.codebook = codebook

        self.converged = False

        self.dist = dist

        self.idxs = idxs

        self.imgs = imgs

        self.norm = norm

        self.info = info

        self._spots = None

    @property
    def spots(self):
        """pd.DataFrame: Lazy-loaded table of detected spots.

        The DataFrame has columns:
        ['barcode_id', 'target', 'mean_intensity', 'max_intensity',
         'area', 'mean_distance', 'min_distance', 'x', 'y']

        Returns:
            pd.DataFrame: One row per detected spot region.
        """
        if self._spots is None:
            self._spots = self._get_spots_df()

        return self._spots

    def _get_spots_df(self):
        """Build a DataFrame of spot properties from decoded indices.

        Returns:
            pd.DataFrame: Spot table with one row per connected region, with columns:
                - barcode_id (int): Index of the barcode.
                - target (str): Name of the barcode target.
                - mean_intensity (float): Mean intensity across the region.
                - max_intensity (float): Maximum intensity within the region.
                - area (int): Pixel count of the region.
                - mean_distance (float): Mean distance from pixels to the assigned barcode vector.
                - min_distance (float): Minimum distance from pixels to the assigned barcode vector.
                - x (float): X-coordinate of the intensity-weighted centroid.
                - y (float): Y-coordinate of the intensity-weighted centroid.
        """
        df = []

        column_names = [
            "barcode_id",
            "target",
            "mean_intensity",
            "max_intensity",
            "area",
            "mean_distance",
            "min_distance",
            "x",
            "y",
        ]

        for i, target in enumerate(self.codebook.targets):
            if target == "background":
                continue

            properties = skimage.measure.regionprops(
                skimage.measure.label(self.idxs == i),
                intensity_image=self.norm,
                cache=False,
            )

            if len(properties) == 0:
                continue

            all_coords = [list(p.coords) for p in properties]

            intensity_and_coords = [np.array([[y[0], y[1], self.norm[y[0], y[1]]] for y in x]) for x in all_coords]

            # Note: Merlin reports norm weighted centroids
            centroid_coords = np.array(
                [
                    (
                        [
                            (r[:, 0] * (r[:, -1] / r[:, -1].sum())).sum(),
                            (r[:, 1] * (r[:, -1] / r[:, -1].sum())).sum(),
                        ]
                        if r.shape[0] > 1
                        else [r[0][0], r[0][1]]
                    )
                    for r in intensity_and_coords
                ]
            )

            intensity_and_areas = np.array([[x[:, 2].mean(), x[:, 2].max(), x.shape[0]] for x in intensity_and_coords])

            centroids = np.zeros((centroid_coords.shape[0], 2))

            centroids[:, [0, 1]] = centroid_coords[:, [1, 0]]

            dist = [[self.dist[y[0], y[1]] for y in x] for x in all_coords]

            df_t = pd.DataFrame(np.zeros((len(properties), len(column_names))), columns=column_names)

            df_t["barcode_id"] = i

            df_t["target"] = target

            df_t.loc[:, ["x", "y"]] = centroids[:, [0, 1]]

            df_t.loc[:, ["mean_intensity", "max_intensity", "area"]] = intensity_and_areas

            df_t["area"] = df_t["area"].astype(int)

            df_t.loc[:, ["mean_distance", "min_distance"]] = np.array(
                [[np.mean(x), np.min(x)] if len(x) > 1 else [x[0], x[0]] for x in dist]
            )

            df.append(df_t)

        if not df:  # Check if the list is empty
            print("Warning: No valid spots were found in the data.")
            return pd.DataFrame(columns=column_names)  # Return an empty DataFrame

        df = pd.concat(df)

        return df


class PixelDecoder(object):
    """Abstract base class for pixel decoding algorithms.

    Defines the interface for `predict`, `fit`, and `score` methods.
    Subclasses **must** override `predict`.

    Attributes:
        codebook (Codebook): The Codebook used for decoding targets.
    """

    # Interface
    def predict(self, imgs):
        """Decode an image stack into a PixelDecoderResult.

        Args:
            imgs (np.ndarray): Array of shape (B, H, W) for B bit planes.

        Returns:
            PixelDecoderResult: Contains decoded indices and spot table.

        Raises:
            NotImplementedError: Always, to enforce override in subclass.
        """
        raise NotImplementedError

    # Optional override if fitting supported
    @property
    def params(self):
        """Returns model parameters after fitting, if supported.

        Returns:
            object or None: Fitted parameters, or None if not applicable.
        """
        return None

    @params.setter
    def params(self, x):
        """Set model parameters; override in subclass to customize.

        Args:
            x (object): Parameter values to assign.
        """
        pass

    def get_update_params(self, local_params):
        """Combine local parameter updates into global parameters.

        Args:
            local_params (list): Per-frame parameter objects.

        Returns:
            object or None: Combined parameters, or None by default.
        """
        return None

    def get_local_update_params(self, imgs):
        """Compute local parameter updates from a single image frame.

        Args:
            imgs (np.ndarray): Single-frame data or stack slice.

        Returns:
            object or None: Local parameter update (None by default).
        """
        return None

    # Optional override if the method has an objective function
    def score(self, img):
        """Score a single image frame for decoding quality.

        Args:
            img (np.ndarray): Single frame of shape (H, W).

        Returns:
            float: Quality score (higher is better; default 0.0).
        """
        return 0

    # Implementation
    def __init__(self, codebook):
        """Initialize a PixelDecoder.

        Args:
            codebook (Codebook): Defines barcode targets and bit mappings.
        """
        self.codebook = codebook

    def fit(self, imgs):
        """Fit the decoder to data by aggregating local updates.

        Args:
            imgs (np.ndarray): Array of frames (B, H, W) to fit on.

        Returns:
            None
        """
        local_params = []

        for x in imgs:
            local_params.append(self.get_local_update_params(x))

        self.params = self.get_update_params(local_params)
