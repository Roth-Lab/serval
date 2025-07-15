"""
Module for aligning bit-planes to a fiducial reference.

Defines FiducialAlignmentImageTransform to apply precomputed
per-frame SimilarityTransforms based on phase cross-correlation.
"""
import cv2
import numpy as np
import skimage

from serval.transform import ImageTransform


class FiducialAlignmentImageTransform(ImageTransform):
    """Aligns images by applying fiducial-based transforms.
    
    Attributes:
        bit_to_round_map (dict): Maps bit->round index.
        fiducial_transforms (list): SimilarityTransforms per round.
        chromatic_corrector (ImageTransform or None): Optional secondary corrector.
    """

    @staticmethod
    def get_fiducial_transforms(imgs, sigma=3, upsample_factor=100):
        """Compute transforms aligning each round to the first round.
        
        Args:
            imgs (ImageStack): Fiducial channel stack.
            sigma (float): Gaussian blur sigma. Defaults to 3.
            upsample_factor (int): Precision for registration. Defaults to 100.
        
        Returns:
            list of SimilarityTransform: One transform per frame.
        """
        filter_size = int(2 * np.ceil(2 * sigma) + 1)

        return _compute_fiducial_transform(imgs, filter_size, sigma, upsample_factor)

    def __init__(self, bit_to_round_map, fiducial_transforms, chromatic_corrector=None):
        """Initialize FiducialAlignmentImageTransform.
        
        Args:
            bit_to_round_map (dict): bit->round index mapping.
            fiducial_transforms (list): Transforms per round.
            chromatic_corrector (ImageTransform, optional): Apply after alignment.
        """
        self.bit_to_round_map = bit_to_round_map

        self.fiducial_transforms = fiducial_transforms

        self.chromatic_corrector = chromatic_corrector

    @property
    def params(self):
        """object or None: Delegated to chromatic_corrector if present."""
        if self.chromatic_corrector is None:
            print("Warning: Chromatic corrector is not initialized.")
            return None
        return self.chromatic_corrector.params

    @params.setter
    def params(self, x):
        """Set parameters on chromatic_corrector if present."""
        self.chromatic_corrector.params = x

    def get_update_params(self, local_params):
        """Delegate parameter updates to chromatic_corrector if present.
        
        Args:
            local_params (list): Local params from get_local_update_params.
        
        Returns:
            object or None: Updated params.
        """
        if self.chromatic_corrector is None:
            update_params = None

        else:
            update_params = self.chromatic_corrector.get_update_params(local_params)

        return update_params

    def get_local_update_params(self, decoded, imgs):
        """Measure offsets via chromatic_corrector if present.
        
        Args:
            decoded (PixelDecoderResult): Decoding result.
            imgs (ImageStack): Raw ImageStack.
        
        Returns:
            object or None: Local params.
        """
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
        """Apply fiducial and chromatic transforms to a frame.
        
        Args:
            img (np.ndarray): 2D image array.
            bit (int): Bit index.
            fov (int): Field-of-view index.
            z (int): Z-slice index.
        
        Returns:
            np.ndarray: Transformed image.
        """
        if self.chromatic_corrector is not None:
            img = self.chromatic_corrector._transform(img, bit, fov, z)

        return skimage.transform.warp(
            img,
            self.fiducial_transforms[fov][self.bit_to_round_map[bit]],
            preserve_range=True,
        )


def _compute_fiducial_transform(imgs, filter_size, sigma, upsample_factor):
    """Compute SimilarityTransforms aligning each round image to the reference frame.
    
    Args:
        imgs (ImageStack): Fiducial channel stack with `.imgs` of shape (frames, H, W).
        filter_size (int): Gaussian kernel size (odd integer) for the pre‑filter.
        sigma (float): Standard deviation of the Gaussian PSF.
        upsample_factor (int): Precision parameter for phase cross‑correlation.
    
    Returns:
        list of SimilarityTransform:
            One transform per frame that maps each image back onto the first (reference) frame.
    """
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
    """Subtract Gaussian-blurred version from image for registration.
    
    Args:
        img (np.ndarray): 2D input image.
        filter_size (int): Gaussian kernel size.
        sigma (float): Gaussian blur sigma.
    
    Returns:
        np.ndarray: High-pass filtered image.
    """
    return img.astype(float) - cv2.GaussianBlur(
        img,
        (filter_size, filter_size),
        sigma,
        borderType=cv2.BORDER_REPLICATE,
    )
