"""
Serial in-memory decoding pipeline module.

Provides a sequential pipeline for testing without parallelism.
"""
from serval.transform import IdentityImageTransform

from .base import DecodingPipeline


class SerialDecodingPipeline(DecodingPipeline):
    """In memory serial decoding pipeline.

    This pipeline does not support any parallelism. It is primarily included for testing.
    
    
    Attributes:
        decoder (PixelDecoder): Underlying pixel decoder.
        max_iters (int): Maximum number of fit iterations. Defaults to 100.
    """

    def __init__(
        self,
        decoder,
        img_transforms,
        max_iters=100,
        transform_preserve_dtype=False,
    ):
        """Initialize a SerialDecodingPipeline.

        Args:
            decoder (PixelDecoder): Underlying pixel decoder.
            img_transforms (list of ImageTransform): Transforms to apply before decoding.
            max_iters (int): Maximum number of fit iterations. Defaults to 100.
            transform_preserve_dtype (bool): If True, preserve dtype of original images after transforms.
        """
        super().__init__(
            decoder, img_transforms, transform_preserve_dtype=transform_preserve_dtype
        )

        self.max_iters = max_iters

    def fit(self, imgs):
        """Fit transformers and decoder sequentially for a fixed number of iterations.
        
        Args:
            imgs (list of ImageStack): Image stacks to use for fitting.
        
        Returns:
            None
        """
        for i in range(self.max_iters):
            print(i + 1)

            decoded = self.predict(imgs)

            self._fit_image_transform(decoded, imgs)

            img_t = self.transform(imgs)

            self.decoder.fit(img_t)

    def predict(self, imgs):
        """Decode each image stack in memory.
        
        Args:
            imgs (list of ImageStack): Image stacks to decode.
        
        Returns:
            list of PixelDecoderResult: Decoded results for each image.
        """
        result = []

        for x in imgs:
            result.append(self.decoder.predict(self._transform_tile(x)))

        return result

    def score(self, imgs):
        """Compute total score across image stacks.
        
        Args:
            imgs (list of ImageStack): Image stacks to score.
        
        Returns:
            float: Sum of decoder scores on each image.
        """
        s = []

        for x in imgs:
            s.append(self.decoder.score(x))

        return sum(s)

    def transform(self, imgs):
        """Apply image transforms sequentially to all stacks.
        
        Args:
            imgs (list of ImageStack): Original image stacks.
        
        Returns:
            list of ImageStack: Transformed image stacks.
        """
        imgs_t = []

        for x in imgs:
            imgs_t.append(self._transform_tile(x))

        return imgs_t

    def _fit_image_transform(self, decoded, imgs):
        """Fit image transform parameters based on decoded results.
        
        Args:
            decoded (list of PixelDecoderResult): Decoding results for current iteration.
            imgs (list of ImageStack): Original image stacks.
        
        Returns:
            None
        """
        imgs_t = imgs

        for t in self.img_transforms:
            new_imgs_t = []

            for x in imgs_t:
                new_imgs_t.append(t.transform(x))

            imgs_t = new_imgs_t

            t.fit(decoded, imgs_t)
