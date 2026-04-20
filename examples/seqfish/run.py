"""Example of how to run a decoding pipeline using the serial pipeline"""

import dask
import dask.dataframe as dd
import h5py
import numpy as np
import pandas as pd
import pathlib
import yaml

from serval.codebook import Codebook
from serval.decode.pixel import CosineOptimizedPixelDecoder, NearestNeigbourPixelDecoder, ScaledImagePixelDecoder
from serval.decode.utils import get_imgs_hist, get_init_scaling_factors
from serval.pipeline import DaskDecodingPipeline
from serval.image import ImageStack

import serval.transform


def main():
    data_dir = pathlib.Path("/home/andrew/projects/serval/data/seqfish_mouse_embryo_serval")

    data_file = data_dir.joinpath("data.h5")

    results_dir = data_dir.joinpath("results")

    spot_file = results_dir.joinpath("scaled.tsv.gz")

    codebook = load_codebook(data_dir.joinpath("codebook.tsv"))

    with open(data_dir.joinpath("bit_to_color.yaml"), "r") as fh:
        bit_to_color_map = yaml.load(fh, yaml.SafeLoader)

    with open(data_dir.joinpath("bit_to_round.yaml"), "r") as fh:
        bit_to_round_map = yaml.load(fh, yaml.SafeLoader)

    penalty_entropy = 1e-2

    penalty_l2 = 1e-3

    results_dir.mkdir(parents=True, exist_ok=True)

    nuclei_imgs = load_nuclei_imgs(data_file)

    fiducial_transforms = load_fiducial_transforms(nuclei_imgs, results_dir)

    spot_imgs = load_spot_imgs(data_file)

    chromatic_corrector_option = serval.transform.ChromaticCorrectionImageTransform(
        bit_to_color_map,
        min_area=5,
        filter_outliers=True,
    )

    img_transforms = [
        serval.transform.FiducialAlignmentImageTransform(
            bit_to_round_map,
            fiducial_transforms,
            chromatic_corrector=chromatic_corrector_option,
        ),
        serval.transform.HighPassImageTransform(sigma=2),
        serval.transform.DeconvoleImageTransform(filter_size=9, num_iters=10, sigma=1.2),
        serval.transform.LowPassImageTransform(sigma=1.0),
    ]

    init_scaling_factors = load_init_scaling_factors(spot_imgs, img_transforms, results_dir)

    nn = NearestNeigbourPixelDecoder(
        codebook,
        max_dist=0.5167,
        min_norm=1,
    )

    decoder = ScaledImagePixelDecoder(
        codebook,
        nn,
        init_scaling_factors=init_scaling_factors,
    )

    # decoder = CosineOptimizedPixelDecoder(
    #     codebook,
    #     nn,
    #     fit_min_area=2,
    #     penalty_entropy=penalty_entropy,
    #     penalty_l2=penalty_l2,
    #     pre_scaling_factors=init_scaling_factors,
    # )

    pipeline = DaskDecodingPipeline(decoder, img_transforms)

    dask.compute(pipeline.fit(list(spot_imgs.values())))

    write_results(spot_file, spot_imgs, pipeline)


def load_codebook(file_name):
    df = pd.read_csv(file_name, index_col="target", sep="\t")

    return Codebook(df)


def load_fiducial_transforms(imgs, results_dir):
    """Load the fiducial transforms infered from bead images.

    This function will cache the fiducial transforms on disk.
    """

    def get_cache_file(fov):
        return cache_dir.joinpath("fov_{}.npy".format(fov))

    def compute_and_save_fiducial_transform(img, out_file):
        fiducial_transform = serval.transform.FiducialAlignmentImageTransform.get_fiducial_transforms(img, sigma=3)

        np.save(out_file, fiducial_transform)

    cache_dir = results_dir.joinpath("fiducial_transforms")

    cache_dir.mkdir(parents=True, exist_ok=True)

    imgs_to_process = []

    for fov in imgs:
        if not get_cache_file(fov).exists():
            imgs_to_process.append(dask.delayed(compute_and_save_fiducial_transform)(imgs[fov], get_cache_file(fov)))

    dask.compute(imgs_to_process)

    fiducial_transforms = {}

    for fov in imgs:
        fiducial_transforms[fov] = np.load(get_cache_file(fov), allow_pickle=True)

    return fiducial_transforms


def load_nuclei_imgs(data_file):
    imgs = {}

    with h5py.File(data_file, "r") as fh:
        for fov in range(3):
            imgs[fov] = ImageStack(
                fh[f"nuclei/fov_{fov}/z_2"][()],
                fov=fov,
            )

    return imgs


def load_spot_imgs(data_file):
    imgs = {}

    with h5py.File(data_file, "r") as fh:
        for fov in range(3):
            for z in [2]:  # range(6):
                imgs[(fov, z)] = ImageStack(
                    fh[f"spots/fov_{fov}/z_{z}"][()],
                    fov=fov,
                    z=z,
                )

    return imgs


def load_init_scaling_factors(imgs, img_transforms, results_dir):
    def get_cache_file(fov, z):
        return cache_dir.joinpath("fov_{fov}-z_{z}.npy".format(fov=fov, z=z))

    def compute_and_save_histograms(img, out_file):
        hist = get_imgs_hist(img, img_transforms)

        np.save(out_file, hist)

    cache_dir = results_dir.joinpath("histograms")

    cache_dir.mkdir(parents=True, exist_ok=True)

    imgs_to_process = []

    for fov, z in imgs:
        if not get_cache_file(fov, z).exists():
            imgs_to_process.append(dask.delayed(compute_and_save_histograms)(imgs[(fov, z)], get_cache_file(fov, z)))

    dask.compute(imgs_to_process)

    keys = list(imgs.keys())

    hists = np.load(get_cache_file(*keys[0]), allow_pickle=True)

    for fov, z in keys[1:]:
        hists += np.load(get_cache_file(fov, z), allow_pickle=True)

    return get_init_scaling_factors(hists)


def write_results(file_name, imgs, pipeline):
    df = []

    for (fov, z), result in zip(imgs.keys(), pipeline.predict(imgs.values())):
        s = result.spots

        s = s.assign(fov=fov, z=z)

        df.append(s)

    spots_df = dd.from_delayed(df)

    spots_df.to_csv(
        file_name,
        compression="gzip",
        index=False,
        single_file=True,
    )


if __name__ == "__main__":
    main()
