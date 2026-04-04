"""
Dask-based decoding pipeline module.

This module provides a parallel, scalable pipeline implementation
using Dask for lazy evaluation and multiprocessing.
"""

import dask
import numpy as np


from .base import DecodingPipeline


class DaskDecodingPipeline(DecodingPipeline):
    """Dask based decoding pipeline.

    This pipeline supports lazy loading and multiprocessing.
    As such it should be scalable to extremely large datasets.


    Attributes:
        decoder (PixelDecoder): Underlying pixel decoder.
        fit_imgs (list of list of ImageStack, optional): Predefined image sets for each iteration.
        fit_imgs_per_iter (int, optional): Number of images to sample per iteration if fit_imgs is None.
        fit_num_iters (int): Total number of fitting iterations. Defaults to 10.
        fit_sample_with_replacement (bool): Whether to sample with replacement when selecting images.
    """

    def __init__(
        self,
        decoder,
        img_transforms,
        fit_imgs=None,
        fit_imgs_per_iter=None,
        fit_num_iters=10,
        fit_sample_with_replacement=False,
        transform_preserve_dtype=False,
    ):
        """Initialize a DaskDecodingPipeline.

        Args:
            decoder (PixelDecoder): Underlying pixel decoder.
            img_transforms (list of ImageTransform): Transforms to apply before decoding.
            fit_imgs (list of list of ImageStack, optional): Predefined image sets for each iteration.
            fit_imgs_per_iter (int, optional): Number of images to sample per iteration if `fit_imgs` is None.
            fit_num_iters (int): Total number of fit iterations. Defaults to 10.
            fit_sample_with_replacement (bool): Whether to sample with replacement when selecting images.
            transform_preserve_dtype (bool): If True, preserve dtype of original images after transforms.
        """
        super().__init__(decoder, img_transforms, transform_preserve_dtype=transform_preserve_dtype)

        self.fit_imgs = fit_imgs

        if fit_imgs is not None:
            self.fit_num_iters = len(fit_imgs)

        else:
            self.fit_num_iters = fit_num_iters

        self.fit_imgs_per_iter = fit_imgs_per_iter

        self.fit_sample_with_replacement = fit_sample_with_replacement

    def fit(self, imgs):
        """Fit image transforms and decoder across multiple Dask-delayed iterations.

        Args:
            imgs (list of ImageStack): Image stacks to use for fitting.
        """
        fit_imgs = self._get_fit_imgs(imgs)

        for i in range(self.fit_num_iters):
            if self.decoder.converged:
                break

            print("Fit iteration: {}".format(i + 1))

            decoded = self.predict(fit_imgs[i])

            self._fit_image_transforms(decoded, fit_imgs[i])

            fit_imgs_t = self.transform(fit_imgs[i])

            self._fit_decoder(fit_imgs_t)

    def predict(self, imgs):
        """Generate Dask-delayed decoding tasks for each image.

        Args:
            imgs (list of ImageStack): Image stacks to decode.

        Returns:
            list of dask.Delayed: Delayed PixelDecoderResult objects.
        """
        result = []

        for x in imgs:
            x = dask.delayed(self._transform_tile)(x)

            result.append(dask.delayed(self.decoder.predict)(x))

        return result

    def score(self, imgs):
        """Generate Dask-delayed scoring tasks and sum the results.

        Args:
            imgs (list of ImageStack): Image stacks to score.

        Returns:
            dask.Delayed: Delayed sum of scores.
        """
        s = []

        for x in imgs:
            x = dask.delayed(self._transform_tile)(x)

            s.append(dask.delayed(self.decoder.score)(x))

        return sum(s)

    def transform(self, imgs):
        """Generate Dask-delayed transform tasks for each image.

        Args:
            imgs (list of ImageStack): Image stacks to transform.

        Returns:
            list of dask.Delayed: Transformed ImageStack objects.
        """
        imgs_t = []

        for x in imgs:
            imgs_t.append(dask.delayed(self._transform_tile)(x))

        return imgs_t

    def _fit_decoder(self, imgs):
        """Fit the decoder parameters using Dask-delayed calls.

        Args:
            imgs (list of dask.Delayed): Transformed image stacks.
        """
        local_params = []

        for x in imgs:
            local_params.append(dask.delayed(self.decoder.get_local_update_params)(x))

        self.decoder.params = dask.delayed(self.decoder.get_update_params)(local_params).compute()

    def _fit_image_transforms(self, decoded, imgs):
        """Fit image transforms parameters using decoded results.

        Args:
            decoded (list of dask.Delayed): Decoding results.
            imgs (list of ImageStack): Original image stacks.
        """
        imgs_t = imgs

        params = []

        for t in self._get_update_transforms():
            # Update the current transform
            local_params = []

            for d, i in zip(decoded, imgs_t):
                local_params.append(dask.delayed(t.get_local_update_params)(d, i))

            params.append(dask.delayed(t.get_update_params)(local_params))

            # Note that the transforms are computed using previous
            # params since those where used to generated decoded
            next_imgs_t = []

            for x in imgs_t:
                next_imgs_t.append(dask.delayed(t.transform)(x))

            imgs_t = next_imgs_t

        # Delayed compute of updated params to let dask see full graph of transformations
        # This saves repeatedly doing the same transform for next_img_t
        params = dask.compute(params)[0]

        transforms = self._get_update_transforms()

        for p, t in zip(params, transforms):
            t.params = p

    def _get_fit_imgs(self, imgs):
        """Determine image subsets for each fit iteration.

        Args:
            imgs (list of ImageStack): All available image stacks.

        Returns:
            list of list of ImageStack: Lists of images per iteration.
        """
        if self.fit_imgs is not None:
            fit_imgs = self.fit_imgs

        elif self.fit_imgs_per_iter is None:
            fit_imgs = [imgs for _ in range(self.fit_num_iters)]

        else:
            imgs_per_iter = self.fit_imgs_per_iter

            if not self.fit_sample_with_replacement:
                imgs_per_iter = min(len(imgs), imgs_per_iter)

            fit_imgs = []

            for i in range(self.fit_num_iters):
                idxs = np.random.choice(
                    len(imgs),
                    replace=self.fit_sample_with_replacement,
                    size=imgs_per_iter,
                )

                fit_imgs.append([imgs[i] for i in idxs])

        return fit_imgs

    def _get_update_transforms(self):
        """Get all transforms up to the last one that supports fit.

        This is a helper to support _fit_image_transforms.
        It uses the fact that we only need to consider param updates up to the last transform that supports them.

        Returns:
            list of ImageTransform: Transforms with non-None `params`.
        """
        update_transforms = []

        update = False

        for t in self.img_transforms[::-1]:
            if t.params is not None:
                update = True

            if update:
                update_transforms.append(t)

        return update_transforms[::-1]
