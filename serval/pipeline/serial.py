from serval.transform import IdentityImageTransform

from .base import DecodingPipeline


class SerialDecodingPipeline(DecodingPipeline):
    """In memory serial decoding pipeline.

    This pipeline does not support any parallelism. It is primarily included for testing.
    """

    def __init__(
        self,
        decoder,
        img_transforms,
        max_iters=100,
        transform_preserve_dtype=False,
    ):
        super().__init__(
            decoder, img_transforms, transform_preserve_dtype=transform_preserve_dtype
        )

        self.max_iters = max_iters

    def fit(self, imgs):
        for i in range(self.max_iters):
            print(i + 1)

            decoded = self.predict(imgs)

            self._fit_image_transform(decoded, imgs)

            img_t = self.transform(imgs)

            self.decoder.fit(img_t)

    def predict(self, imgs):
        result = []

        for x in imgs:
            result.append(self.decoder.predict(self._transform_tile(x)))

        return result

    def score(self, imgs):
        s = []

        for x in imgs:
            s.append(self.decoder.score(x))

        return sum(s)

    def transform(self, imgs):
        imgs_t = []

        for x in imgs:
            imgs_t.append(self._transform_tile(x))

        return imgs_t

    def _fit_image_transform(self, decoded, imgs):
        imgs_t = imgs

        for t in self.img_transforms:
            new_imgs_t = []

            for x in imgs_t:
                new_imgs_t.append(t.transform(x))

            imgs_t = new_imgs_t

            t.fit(decoded, imgs_t)
