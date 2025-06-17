import numpy as np
import skimage

from serval.decode.pixel import PixelDecoder
from serval.image import ImageStack


class ScaledImagePixelDecoder(PixelDecoder):
    """Scale image by factors estimated from decoded spots.

    This code is largely based off of https://github.com/emanuega/MERlin/blob/master/merlin/analysis/optimize.py
    """

    def __init__(self, codebook, decoder, init_scaling_factors=None, min_area=3):
        super().__init__(codebook)

        self.decoder = decoder

        self.min_area = min_area

        if init_scaling_factors is None:
            init_scaling_factors = np.ones(codebook.num_bits)

        self.scaling_factors = init_scaling_factors

    @property
    def params(self):
        return self.scaling_factors

    @params.setter
    def params(self, x):
        self.scaling_factors = x

    # Override for fit interface
    def get_update_params(self, local_params):
        refactors = np.stack(local_params, axis=0)

        refactors[refactors == 0] = 1

        return np.nanmedian(self.scaling_factors * refactors, axis=0)

    def get_local_update_params(self, imgs):
        decoded = self.predict(imgs)

        unscaled_pixel_trace = decoded.info["X"] * decoded.norm

        return self._get_refactors(decoded.idxs, unscaled_pixel_trace)

    # Interface
    def predict(self, imgs):
        return self.decoder.predict(self._get_scaled_img(imgs))

    # Helper methods
    def _get_scaled_img(self, imgs):
        return ImageStack(
            imgs.imgs / self.scaling_factors[:, np.newaxis, np.newaxis],
            imgs.fov,
            imgs.z,
        )

    def _get_refactors(self, decoded_pixels, pixel_trace):
        # Currently we don't estimate but here for future compatibility
        background_refactors = np.zeros(self.codebook.num_bits)

        sum_pixel_traces = np.zeros((self.codebook.num_targets, self.codebook.num_bits))

        barcodes_seen = np.zeros(self.codebook.num_targets)

        for b in range(self.codebook.num_targets):
            barcode_regions = [
                x
                for x in skimage.measure.regionprops(
                    skimage.measure.label((decoded_pixels == b).astype(int))
                )
                if x.area >= self.min_area
            ]

            barcodes_seen[b] = len(barcode_regions)

            for br in barcode_regions:
                mean_pixel_trace = np.mean(
                    [pixel_trace[:, y[0], y[1]] for y in br.coords],
                    axis=0,
                )

                mean_pixel_trace -= background_refactors

                norm_pixel_trace = mean_pixel_trace / np.linalg.norm(mean_pixel_trace)

                sum_pixel_traces[b, :] += norm_pixel_trace / barcodes_seen[b]

        sum_pixel_traces[self.codebook.barcode_matrix == 0] = np.nan

        on_bit_intensity = np.nanmean(sum_pixel_traces, axis=0)

        return on_bit_intensity / np.mean(on_bit_intensity)
