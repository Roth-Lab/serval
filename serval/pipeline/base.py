"""
Decoding pipelines: Abstract base class for decoding image stacks.

This module defines the DecodingPipeline interface for fitting,
transforming, predicting, and scoring image stacks using pixel decoders.
"""

from serval.image.stack import ImageStack
from serval.transform import IdentityImageTransform


class DecodingPipeline(object):
    """Abstract base class for decoding a set of images.

    This class defines the common interface for image decoding pipelines,
    including fitting transformers and decoders, applying transforms,
    and scoring predictions in serial or parallel contexts.


    Attributes:
        decoder (PixelDecoder): The underlying pixel decoder.
        img_transforms (list): Sequence of image transforms.
        transform_preserve_dtype (bool): Whether to enforce original dtype on transformed images.
    """

    # Interface
    def fit(self, imgs):
        """Takes a list of ImageStack objects and fits the transformations and decoder

        Args:
            imgs (list of ImageStack): Image stacks to use for fitting.
        """
        raise NotImplementedError

    def predict(self, imgs):
        """Takes a list of ImageStack objects and returns a list of PixelDecoderResult

        Args:
            imgs (list of ImageStack): Image stacks to decode.

        Returns:
            list of PixelDecoderResult: Decoded results for each image.
        """
        raise NotImplementedError

    def score(self, imgs):
        """Takes a list of ImageStack objects and returns the total score across all images

        Args:
            imgs (list of ImageStack): Image stacks to score.

        Returns:
            float: Sum of decoder scores on each image.
        """
        raise NotImplementedError

    def transform(self, imgs):
        """Takes a list of ImageStack objects transforms them to a new list of ImageStack objects

        Args:
            imgs (list of ImageStack): Original image stacks.

        Returns:
            list of ImageStack: Transformed image stacks.
        """
        raise NotImplementedError

    # Implementation
    def __init__(
        self,
        decoder,
        img_transforms,
        transform_preserve_dtype=False,
    ):
        """Initialize a DecodingPipeline.

        Args:
            decoder (PixelDecoder): Pixel-based decoder implementing `predict`, `fit`, and `score`.
            img_transforms (list of ImageTransform): List of transforms applied before decoding.
            transform_preserve_dtype (bool): If True, preserve dtype of original images after transforms.
        """
        self.decoder = decoder

        if img_transforms is None:
            img_transforms = [IdentityImageTransform()]

        self.img_transforms = img_transforms

        self.transform_preserve_dtype = transform_preserve_dtype

    def _transform_tile(self, img):
        """Apply sequential image transforms to a single tile.

        Args:
            img (ImageStack): Input image stack.

        Returns:
            ImageStack: Transformed image stack, optionally preserving dtype.
        """
        img_t = img

        for t in self.img_transforms:
            img_t = t.transform(img_t, preserve_dtype=False)

        if self.transform_preserve_dtype:
            img_t = ImageStack(img_t.imgs.astype(img.imgs.dtype), fov=img_t.fov, z=img_t.z)

        return img_t
