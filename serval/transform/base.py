import numpy as np

from serval.image import ImageStack


class ImageTransform(object):
    # Interface
    def _transform(self, img, bit, fov, z):
        """Transform a 2D image for a given fov, z-slice and bit."""
        raise NotImplemented

    # Optional override if fitting supported
    @property
    def params(self):
        return None

    @params.setter
    def params(self, x):
        pass

    def get_update_params(self, local_params):
        """Compute the updated set of global parameters given a list of local parameters"""
        return None

    def get_local_update_params(self, decoded, imgs):
        """Compute the local parameters to update the transform given a single PixelDecoderResult and ImageStack"""
        return None

    # Implementation
    def fit(self, decoded, imgs):
        """Fit the transform given a list of PixelDecoderResult and ImageStacks"""
        local_params = []

        for d, i in zip(decoded, imgs):
            local_params.append(self.get_local_update_params(d, i))

        self.params = self.get_update_params(local_params)

    def transform(self, imgs, preserve_dtype=True):
        """Transform an ImageStack"""
        imgs_t = []

        for bit in range(imgs.num_frames):
            x = self._transform(imgs.imgs[bit], bit, imgs.fov, imgs.z)

            if preserve_dtype:
                x = x.astype(imgs.imgs[0].dtype)

            imgs_t.append(x)

        return ImageStack(np.stack(imgs_t, axis=0), fov=imgs.fov, z=imgs.z)


class IdentityImageTransform(ImageTransform):
    """Image transform that return original image"""

    def _transform(self, img, bit, fov, z):
        return img
