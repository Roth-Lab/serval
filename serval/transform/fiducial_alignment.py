import cv2
import numpy as np
import skimage

from serval.transform import ImageTransform


class FiducialAlignmentImageTransform(ImageTransform):
    @staticmethod
    def get_fiducial_transforms(imgs, sigma=3, upsample_factor=100):
        filter_size = int(2 * np.ceil(2 * sigma) + 1)

        return _compute_fiducial_transform(imgs, filter_size, sigma, upsample_factor)

    def __init__(self, bit_to_round_map, fiducial_transforms, chromatic_corrector=None):
        self.bit_to_round_map = bit_to_round_map

        self.fiducial_transforms = fiducial_transforms

        self.chromatic_corrector = chromatic_corrector

    @property
    def params(self):
        if self.chromatic_corrector is None:
            print("Warning: Chromatic corrector is not initialized.")
            return None
        return self.chromatic_corrector.params

    @params.setter
    def params(self, x):
        self.chromatic_corrector.params = x

    def get_update_params(self, local_params):
        if self.chromatic_corrector is None:
            update_params = None

        else:
            update_params = self.chromatic_corrector.get_update_params(local_params)

        return update_params

    def get_local_update_params(self, decoded, imgs):
        if self.chromatic_corrector is None:
            local_params = None

        else:
            imgs = self.transform(imgs)

            local_params = self.chromatic_corrector.get_offsets(
                decoded, imgs, debug=False
            )

        return local_params

    # Interface
    def _transform(self, img, bit, fov, z):
        if self.chromatic_corrector is not None:
            img = self.chromatic_corrector._transform(img, bit, fov, z)

        return skimage.transform.warp(
            img,
            self.fiducial_transforms[fov][self.bit_to_round_map[bit]],
            preserve_range=True,
        )


def _compute_fiducial_transform(imgs, filter_size, sigma, upsample_factor):
    fixed_img = _filter(imgs.imgs[0], filter_size, sigma)

    offsets = []

    for r in range(imgs.num_frames):
        offsets.append(
            skimage.registration.phase_cross_correlation(
                fixed_img,
                _filter(imgs.imgs[r], filter_size, sigma),
                normalization=None,
                upsample_factor=upsample_factor,
            )[0]
        )

    return [
        skimage.transform.SimilarityTransform(translation=[-x[1], -x[0]])
        for x in offsets
    ]


def _filter(img, filter_size, sigma):
    return img.astype(float) - cv2.GaussianBlur(
        img,
        (filter_size, filter_size),
        sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
