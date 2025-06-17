from serval.image.stack import ImageStack
from serval.transform import IdentityImageTransform


class DecodingPipeline(object):
    """Abstract class for decoding a set of images

    Parameters
    ----------
    decoder: (PixelDecoder like) Pixel based decoder
    img_transforms: (list) List of ImageTransforms to apply sequentially
    """

    # Interface
    def fit(self, imgs):
        """Takes a list of ImageStack objects and fits the transformations and decoder"""
        raise NotImplementedError

    def predict(self, imgs):
        """Takes a list of ImageStack objects and returns a list of PixelDecoderResult"""
        raise NotImplementedError

    def score(self, imgs):
        """Takes a list of ImageStack objects and returns the total score across all images"""

    def transform(self, imgs):
        """Takes a list of ImageStack objects transforms them to a new list of ImageStack objects"""
        raise NotImplementedError

    # Implementation
    def __init__(
        self,
        decoder,
        img_transforms,
        transform_preserve_dtype=False,
    ):
        self.decoder = decoder

        if img_transforms is None:
            img_transforms = [IdentityImageTransform()]

        self.img_transforms = img_transforms

        self.transform_preserve_dtype = transform_preserve_dtype

    def _transform_tile(self, img):
        img_t = img

        for t in self.img_transforms:
            img_t = t.transform(img_t, preserve_dtype=False)

        if self.transform_preserve_dtype:
            img_t = ImageStack(
                img_t.imgs.astype(img.imgs.dtype), fov=img_t.fov, z=img_t.z
            )

        return img_t
