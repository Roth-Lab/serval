import numpy as np
import pandas as pd
import skimage


class PixelDecoderResult(object):
    def __init__(self, codebook, dist, idxs, imgs, norm, info=None):
        self.codebook = codebook

        self.dist = dist

        self.idxs = idxs

        self.imgs = imgs

        self.norm = norm

        self.info = info

        self._spots = None

    @property
    def spots(self):
        if self._spots is None:
            self._spots = self._get_spots_df()

        return self._spots

    def _get_spots_df(self):
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

            intensity_and_coords = [
                np.array([[y[0], y[1], self.norm[y[0], y[1]]] for y in x])
                for x in all_coords
            ]

            # Note: Merlin reports norm weighted centroids
            centroid_coords = np.array(
                [
                    [
                        (r[:, 0] * (r[:, -1] / r[:, -1].sum())).sum(),
                        (r[:, 1] * (r[:, -1] / r[:, -1].sum())).sum(),
                    ]
                    if r.shape[0] > 1
                    else [r[0][0], r[0][1]]
                    for r in intensity_and_coords
                ]
            )

            intensity_and_areas = np.array(
                [
                    [x[:, 2].mean(), x[:, 2].max(), x.shape[0]]
                    for x in intensity_and_coords
                ]
            )

            centroids = np.zeros((centroid_coords.shape[0], 2))

            centroids[:, [0, 1]] = centroid_coords[:, [1, 0]]

            dist = [[self.dist[y[0], y[1]] for y in x] for x in all_coords]

            df_t = pd.DataFrame(
                np.zeros((len(properties), len(column_names))), columns=column_names
            )

            df_t["barcode_id"] = i

            df_t["target"] = target

            df_t.loc[:, ["x", "y"]] = centroids[:, [0, 1]]

            df_t.loc[
                :, ["mean_intensity", "max_intensity", "area"]
            ] = intensity_and_areas

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
    # Interface
    def predict(self, imgs):
        raise NotImplementedError

    # Optional override if fitting supported
    @property
    def params(self):
        return None

    @params.setter
    def params(self, x):
        pass

    def get_update_params(self, local_params):
        return None

    def get_local_update_params(self, imgs):
        return None

    # Optional override if the method has an objective function
    def score(self, img):
        return 0

    # Implementation
    def __init__(self, codebook):
        self.codebook = codebook

    def fit(self, imgs):
        local_params = []

        for x in imgs:
            local_params.append(self.get_local_update_params(x))

        self.params = self.get_update_params(local_params)
