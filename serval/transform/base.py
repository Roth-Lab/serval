"""
Module for core image transforms: interface and identity.

Defines:
- ImageTransform: abstract base for ImageStack transforms.
- IdentityImageTransform: pass-through transform.
"""
import numpy as np

from serval.image import ImageStack


class ImageTransform(object):
    """Abstract base class for image stack transformations."""
    
    # Interface
    def _transform(self, img, bit, fov, z):
        """Transform a 2D image for a given fov, z-slice and bit.
    
        Args:
            img (np.ndarray): 2D image array.
            bit (int): Bit index of this frame.
            fov (int): Field-of-view index.
            z (int): Z-slice index.
        
        Returns:
            np.ndarray: Transformed 2D image.
        
        Raises:
            NotImplementedError: If not overridden in subclass.
        """
        raise NotImplementedError

    # Optional override if fitting supported
    @property
    def params(self):
        """object or None: Current transform parameters."""
        return None

    @params.setter
    def params(self, x):
        """Set transform parameters.
        
        Args:
            x (object): New transformation parameters.
        """
        pass

    def get_update_params(self, local_params):
        """Compute the updated set of global parameters given a list of local parameters
        
        Args:
            local_params (list): Local parameter updates.
        
        Returns:
            object or None: Updated global parameters.
        """
        return None

    def get_local_update_params(self, decoded, imgs):
        """Compute the local parameters to update the transform given a single PixelDecoderResult and ImageStack
        
        Args:
            decoded (PixelDecoderResult): Decoding result.
            imgs (ImageStack): Corresponding ImageStack.

        Returns:
            object or None: Local parameter update.
        """
        return None

    # Implementation
    def fit(self, decoded, imgs):
        """Fit the transform given a list of PixelDecoderResult and ImageStacks
        
        Args:
            decoded (list of PixelDecoderResult): Decoding results.
            imgs (list of ImageStack): Corresponding image stacks.
        """
        local_params = []

        for d, i in zip(decoded, imgs):
            local_params.append(self.get_local_update_params(d, i))

        self.params = self.get_update_params(local_params)

    def transform(self, imgs, preserve_dtype=True):
        """Transform an ImageStack
        
        Args:
            imgs (ImageStack): Input ImageStack.
            preserve_dtype (bool): If True, cast outputs to original dtype.
        
        Returns:
            ImageStack: Transformed ImageStack.
        """
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
        """Return the original image without modification."""
        return img
