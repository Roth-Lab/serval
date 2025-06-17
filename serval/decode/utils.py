import numpy as np
import skimage.util


def crop_image(img, crop_size=0):
    if crop_size > 0:
        img_new = []

        for i in range(img.shape[0]):
            img_new.append(skimage.util.crop(img[i], crop_size))

        img = np.array(img_new)

    return img


def normalize(x):
    norm = np.linalg.norm(x, axis=1)

    norm[norm == 0] = 1

    return x / norm[:, np.newaxis]


def reshape_data(imgs):
    return imgs.reshape((imgs.shape[0], np.prod(imgs.shape[1:]))).T


def get_init_scaling_factors(hists, q=0.9):
    """Given pixel intensity histograms computed across each bit compute initial scaling factors

    Basically computes the quantile.
    """
    num_bits = hists.shape[0]

    scaling_factors = np.zeros(num_bits)

    for b in range(num_bits):
        cum_dist = hists[b].cumsum()

        prob_dist = cum_dist / cum_dist[-1]

        # Note: This should probably be +1 but +2 matches MERLIn
        scaling_factors[b] = np.argmin(np.abs(prob_dist - q)) + 2

    return scaling_factors


def get_imgs_hist(imgs, transforms):
    """Compute pixel histograms for each round"""
    for t in transforms:
        imgs = t.transform(imgs)

    bins = np.arange(np.iinfo(np.uint16).max)

    hist = []

    for i in range(imgs.num_frames):
        hist.append(np.histogram(imgs.imgs[i], bins=bins)[0])

    return np.stack(hist, axis=0)
