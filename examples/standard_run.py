import pathlib
import pickle
import json
import dask
import sys

dask.config.set({"dataframe.query-planning": True})

import dask.dataframe as dd
import numpy as np


from serval.decode.pixel import (
    CosineOptimizedPixelDecoder,
    NearestNeigbourPixelDecoder,
    ScaledImagePixelDecoder,
)
from serval.decode.utils import get_imgs_hist, get_init_scaling_factors
from serval.image import ImageStack
from serval.pipeline import DaskDecodingPipeline

import serval.io.merlin
import serval.transform

import os
from datetime import datetime
import ray
import argparse

from ray.util.dask import enable_dask_on_ray


def load_config(config_file):
    with open(config_file, "r") as file:
        config = json.load(file)
    return config


def main(config):
    decoder_type = config["decoder_type"]

    filter_outliers = config["filter_outliers"]

    cosine_penalty = config["cosine_penalty"]

    # Set to true to match the exact tiles used by MERLIN for optimization
    use_merlin_opt_tiles = config["use_merlin_opt_tiles"]

    # If not using MERLIN we can specify the number of images to use at each round of optimization
    # NOTE: Set to None to use whole dataset
    # NOTE: This has no effect if the use_merlin_opt_tiles flag is True
    fit_imgs_per_iter = config["fit_imgs_per_iter"]

    if not use_merlin_opt_tiles:
        print(
            "Warning: To compare decoding method with Merlin, it is suggested to use Merlin Optimization Tiles."
        )

    if decoder_type == "scaled" and filter_outliers:
        print(
            "Error: Serval Merlin (scaled decoder) requires filter outliers set to False."
        )
        sys.exit(1)

    if decoder_type == "cosine" and not filter_outliers:
        print("Warning: It is suggested to use filter outliers for Cosine decoder.")

    # Load up inputs from merlin run
    merlin_results_dir = pathlib.Path(config["merlin_results_dir"])

    codebook_file = merlin_results_dir.joinpath(config["codebook_file"])

    data_org_file = merlin_results_dir.joinpath(config["data_org_file"])

    img_dir = pathlib.Path(config["img_dir"])

    # Specify where serval results go
    results_dir = pathlib.Path(config["results_dir"])

    results_dir.mkdir(parents=True, exist_ok=True)

    # Load up the data and metadata
    dataset = serval.io.merlin.MerlinDataset(codebook_file, data_org_file, img_dir)

    # Load images for experiment
    bead_imgs, spot_imgs = load_dataset_imgs(dataset)

    # Load the fiducial transforms
    fiducial_transforms = load_fiducial_transforms(bead_imgs, results_dir)

    # Conditional setting of chromatic_corrector_option
    if decoder_type == "simple":
        chromatic_corrector_option = None
    else:
        chromatic_corrector_option = serval.transform.ChromaticCorrectionImageTransform(
            dataset.data_org.bit_to_color_map,
            min_area=5,
            filter_outliers=filter_outliers,
        )

    # Setup image transformations
    img_transforms = [
        serval.transform.FiducialAlignmentImageTransform(
            dataset.data_org.bit_to_round_map,
            fiducial_transforms,
            chromatic_corrector=chromatic_corrector_option,
        ),
        serval.transform.HighPassImageTransform(sigma=2),
        serval.transform.DeconvoleImageTransform(
            filter_size=9, num_iters=10, sigma=1.2
        ),
        serval.transform.LowPassImageTransform(sigma=1.0),
    ]

    # Load initial scaling factors
    init_scaling_factors = load_init_scaling_factors(
        spot_imgs, img_transforms, results_dir
    )

    # We added this line on May 22, 2024 since this is what the original Merlin does.
    nn = NearestNeigbourPixelDecoder(
        dataset.codebook,
        max_dist=0.5176,
        min_norm=1,
    )

    # Choose which type of decoder to use and initialize
    if decoder_type != "simple":
        if decoder_type == "scaled":
            decoder = ScaledImagePixelDecoder(
                dataset.codebook,
                nn,
                init_scaling_factors=init_scaling_factors,
                min_area=5,
            )

        elif decoder_type == "cosine":
            decoder = CosineOptimizedPixelDecoder(
                dataset.codebook,
                nn,
                fit_max_size=int(1e5),
                fit_min_area=5,
                init_scaling_factors=init_scaling_factors
                / np.mean(init_scaling_factors),
                penalty=int(cosine_penalty),
            )

        print(decoder.scaling_factors)

    elif decoder_type == "simple":
        decoder = nn

    else:
        print("Error: Invalid decoder type.")
        sys.exit(1)

    # Decide whether to use same tiles as MERLIN for optimization
    if use_merlin_opt_tiles:
        fit_imgs = load_merlin_fit_images(spot_imgs, merlin_results_dir)
    else:
        fit_imgs = None

    pipeline = DaskDecodingPipeline(
        decoder, img_transforms, fit_imgs=fit_imgs, fit_imgs_per_iter=fit_imgs_per_iter
    )

    if decoder_type != "simple":
        dask.compute(pipeline.fit(list(spot_imgs.values())))

        print(decoder.scaling_factors)

    # Save the trained decoder and transforms in case we need it later
    out_file = results_dir.joinpath("{}_fit.pkl".format(decoder_type))

    with open(out_file, "wb") as fh:
        pickle.dump({"decoder": decoder, "transforms": img_transforms}, fh)

    # To match MERLIN behaviour change the final low pass filter from sigma=1.0 used for training to sigma=0.6
    # for final decoding
    img_transforms.pop()

    img_transforms.append(serval.transform.LowPassImageTransform(sigma=0.6))
    
    # Update the decoder for final decoding
    decoder.decoder = NearestNeigbourPixelDecoder(
        dataset.codebook,
        max_dist=0.5167,
        min_norm=1,
    )

    pipeline = DaskDecodingPipeline(decoder, img_transforms)

    # TODO: To match the "Decode" task of Merlin we should support cropping here
    # Finally decode and save results
    spots_df = []

    for (fov, z), result in zip(
        spot_imgs.keys(), pipeline.predict(list(spot_imgs.values()))
    ):
        s = result.spots

        s = s.assign(fov=fov, z=z)

        spots_df.append(s)

    spots_df = dd.from_delayed(spots_df)

    out_file = results_dir.joinpath("{}_spots.csv.gz".format(decoder_type))

    spots_df.to_csv(
        out_file,
        compression="gzip",
        index=False,
        single_file=True,
    )


def load_dataset_imgs(dataset):
    """Returns a pair of dictionaries for bead and spot images for the dataset.

    Returns
    -------
    bead_imgs: (dict) Keys are fov and values dask delayed images for fiducial beads
    spots_imgs: (dict) Keys are (fov, z) and values dask delayed images for spots
    """

    def load_bead_stack(fov):
        return ImageStack(dataset.get_fiducial_image_stack(fov), fov=fov)

    def load_spot_stack(fov, z):
        return ImageStack(dataset.get_primary_image_stack(fov, z), fov=fov, z=z)

    bead_imgs = {}

    for fov in dataset.data_org.fovs:
        bead_imgs[fov] = dask.delayed(load_bead_stack)(fov)

    spot_imgs = {}

    for fov in dataset.data_org.fovs:
        for z in range(dataset.data_org.num_z_slices):
            spot_imgs[(fov, z)] = dask.delayed(load_spot_stack)(fov, z)

    return bead_imgs, spot_imgs


def load_fiducial_transforms(imgs, results_dir):
    """Load the fiducial transforms infered from bead images.

    This function will cache the fiducial transforms on disk.
    """

    def get_cache_file(fov):
        return cache_dir.joinpath("fov_{}.npy".format(fov))

    def compute_and_save_fiducial_transform(img, out_file):
        fiducial_transform = (
            serval.transform.FiducialAlignmentImageTransform.get_fiducial_transforms(
                img, sigma=3
            )
        )

        np.save(out_file, fiducial_transform)

    cache_dir = results_dir.joinpath("fiducial_transforms")

    cache_dir.mkdir(parents=True, exist_ok=True)

    imgs_to_process = []

    for fov in imgs:
        if not get_cache_file(fov).exists():
            imgs_to_process.append(
                dask.delayed(compute_and_save_fiducial_transform)(
                    imgs[fov], get_cache_file(fov)
                )
            )

    dask.compute(imgs_to_process)

    fiducial_transforms = {}

    for fov in imgs:
        fiducial_transforms[fov] = np.load(get_cache_file(fov), allow_pickle=True)

    return fiducial_transforms


def load_init_scaling_factors(imgs, img_transforms, results_dir):
    """Load the initial scaling factors based on the transformed spots images.

    This function will cache the pixel intensity histograms on disk.
    """

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
            imgs_to_process.append(
                dask.delayed(compute_and_save_histograms)(
                    imgs[(fov, z)], get_cache_file(fov, z)
                )
            )

    dask.compute(imgs_to_process)

    keys = list(imgs.keys())

    hists = np.load(get_cache_file(*keys[0]), allow_pickle=True)

    for fov, z in keys[1:]:
        hists += np.load(get_cache_file(fov, z), allow_pickle=True)

    return get_init_scaling_factors(hists)


def load_merlin_fit_images(imgs, merlin_results_dir):
    def get_num_iters():
        return len(list(merlin_results_dir.glob("Optimize*")))

    def get_num_tiles_per_iter():
        return len(
            list(merlin_results_dir.joinpath("Optimize1").glob("select_frame_*.npy"))
        )

    num_iters = get_num_iters()

    num_tiles_per_iter = get_num_tiles_per_iter()

    fit_imgs = []

    for i in range(num_iters):
        round_imgs = []

        for j in range(num_tiles_per_iter):
            file_name = merlin_results_dir.joinpath(
                "Optimize{}".format(i + 1), "select_frame_{}.npy".format(j)
            )

            fov, z = np.load(file_name)

            round_imgs.append(imgs[(fov, z)])

        fit_imgs.append(round_imgs)

    # print(len(fit_imgs), len(fit_imgs[0]))

    return fit_imgs


def execute_with_ray(config, default_cpus=100, default_mem=250):
    # Generate a timestamp
    timestamp = datetime.now().strftime("%S%H%M%S")

    # temporary directory path
    temp_dir = os.path.normpath(
        os.path.join("/output_dir", timestamp, "ray")
    )
    object_spilling_dir = os.path.normpath(
        os.path.join("/output_dir", timestamp, "ray_objspill")
    )

    # Request 8 cores and 128GB of memory
    ray.init(
        num_cpus=default_cpus,
        object_store_memory=default_mem * 10**9,
        _temp_dir=temp_dir,  # General temp directory
        _system_config={
            "local_fs_capacity_threshold": 1,
            "object_spilling_config": json.dumps(
                {
                    "type": "filesystem",
                    "params": {
                        "directory_path": object_spilling_dir,  # Specific for object spilling
                    },
                }
            ),
        },
    )

    enable_dask_on_ray()

    main(config)

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the main script with a configuration file."
    )
    parser.add_argument(
        "config_file", type=str, help="Path to the JSON configuration file."
    )
    args = parser.parse_args()

    config = load_config(args.config_file)
    execute_with_ray(config)
