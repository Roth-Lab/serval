"""
Module for loading and managing multi-frame image stacks with associated metadata.
"""

import skimage


class ImageStack(object):
    """A multi-frame image stack with field-of-view and depth metadata.

    This class wraps a stack of images (frames) loaded from disk and keeps track of
    which field-of-view (FOV) and Z-slice depth the stack corresponds to.

    Attributes:
        file_name (str): Path to the image file to load (e.g., TIF, TIFF, PNG).
        fov (int): Field-of-view index associated with this stack. Defaults to 0.
        z (int): Z-slice index associated with this stack. Defaults to 0.
    """

    @staticmethod
    def load(file_name, fov=0, z=0):
        """Load an image stack from a file and return an ImageStack instance.

        Args:
            file_name (str): Path to the image file to load (e.g., TIF, TIFF, PNG).
            fov (int): Field-of-view index associated with this stack. Defaults to 0.
            z (int): Z-slice index associated with this stack. Defaults to 0.

        Returns:
            ImageStack: A new instance containing the loaded image stack and metadata.
        """
        return ImageStack(skimage.io.imread(file_name), fov=fov, z=z)

    def __init__(self, imgs, fov=0, z=0):
        """Initialize an ImageStack.

        Args:
            imgs (np.ndarray): A 3D array of image frames (frames × height × width).
            fov (int): Field-of-view index associated with this stack. Defaults to 0.
            z (int): Z-slice index associated with this stack. Defaults to 0.
        """
        self.imgs = imgs

        self.fov = fov

        self.z = z

    @property
    def num_frames(self):
        """Number of frames in the image stack.

        Returns:
            int: The number of images (frames) in the stack (dimension 0 of `imgs`).
        """
        return self.imgs.shape[0]
