"""Example of how to run a decoding pipeline using the serial pipeline"""

import dask.dataframe as dd
import h5py
import numpy as np
import pandas as pd
import pathlib
import pickle

from serval.codebook import Codebook
from serval.decode.pixel import CosineOptimizedPixelDecoder, NearestNeigbourPixelDecoder, ScaledImagePixelDecoder
from serval.pipeline import DaskDecodingPipeline
from serval.image import ImageStack


def main():
    codebook_file = "codebook.tsv"

    data_file = "data.h5"

    out_dir = pathlib.Path("results")

    penalty_entropy = 1e-2

    penalty_l2 = 1e-3

    out_dir.mkdir(parents=True, exist_ok=True)

    codebook_no_blanks = load_codebook(codebook_file, drop_blanks=True)

    imgs = load_data(data_file)

    init_scaling_factors = load_init_scaling_factors(imgs)

    nn_decoder = NearestNeigbourPixelDecoder(codebook_no_blanks, max_dist=0.3321, min_norm=0.4)

    print("#" * 100)
    print("Step 1: Initial scaling factors")

    out_file = out_dir.joinpath("scaled_init.tsv.gz")

    if not out_file.exists():
        decoder = ScaledImagePixelDecoder(
            codebook_no_blanks,
            nn_decoder,
            init_scaling_factors=init_scaling_factors,
        )

        pipeline = DaskDecodingPipeline(decoder, [], fit_num_iters=5)

        write_results(codebook_file, out_file, imgs, pipeline)

    print()
    print("#" * 100)
    print("Step 2: Fitted scaled decoder")

    out_file = out_dir.joinpath("scaled_fit.tsv.gz")

    if not out_file.exists():
        decoder = ScaledImagePixelDecoder(
            codebook_no_blanks,
            nn_decoder,
            init_scaling_factors=init_scaling_factors,
            min_area=2,
        )

        pipeline = DaskDecodingPipeline(decoder, [], fit_num_iters=10)

        pipeline.fit(imgs)

        write_results(codebook_file, out_file, imgs, pipeline)

        out_file = out_dir.joinpath("scaled_decoder.pkl")

        with open(out_file, "wb") as fh:
            pickle.dump({"decoder": decoder}, fh)

    print()
    print("#" * 100)
    print("Step 3: Fitted cosine decoder using initial scaling factors")

    out_file = out_dir.joinpath("cosine_init_fit.tsv.gz")

    if not out_file.exists():
        decoder = CosineOptimizedPixelDecoder(
            codebook_no_blanks,
            nn_decoder,
            fit_min_area=2,
            penalty_entropy=penalty_entropy,
            penalty_l2=penalty_l2,
            pre_scaling_factors=init_scaling_factors,
        )

        pipeline = DaskDecodingPipeline(decoder, [], fit_num_iters=10)

        pipeline.fit(imgs)

        write_results(codebook_file, out_file, imgs, pipeline)

        out_file = out_dir.joinpath("cosine_init.pkl")

        with open(out_file, "wb") as fh:
            pickle.dump({"decoder": decoder}, fh)

    print()
    print("#" * 100)
    print("Step 4: Fitted cosine decoder using scaled decoder scaling factors")

    out_file = out_dir.joinpath("cosine_scaled_fit.tsv.gz")

    if not out_file.exists():
        file_name = out_dir.joinpath("scaled_decoder.pkl")

        with open(file_name, "rb") as fh:
            result = pickle.load(fh)

        decoder = CosineOptimizedPixelDecoder(
            codebook_no_blanks,
            nn_decoder,
            fit_min_area=2,
            penalty_entropy=penalty_entropy,
            penalty_l2=penalty_l2,
            pre_scaling_factors=result["decoder"].scaling_factors,
        )

        pipeline = DaskDecodingPipeline(decoder, [], fit_num_iters=10)

        pipeline.fit(imgs)

        write_results(codebook_file, out_file, imgs, pipeline)

        out_file = out_dir.joinpath("cosine_scaled.pkl")

        with open(out_file, "wb") as fh:
            pickle.dump({"decoder": decoder}, fh)


def load_codebook(file_name, drop_blanks=True):
    df = pd.read_csv(file_name, sep="\t")

    df = df.rename(columns={"gene_names": "target"}).set_index("target")

    df = df.drop(columns="FPKM_data")

    df = df.rename(columns=lambda x: f"bit_{x}")

    if drop_blanks:
        df = df[~df.index.str.startswith("Blank")]

    return Codebook(df)


def load_data(data_file, img_stage="filtered_clipped"):
    imgs = []

    with h5py.File(data_file, "r") as fh:
        for fov in range(4):
            fov_grp = fh[f"fov_{fov}"]

            fov_stack = fov_grp[img_stage][()]

            imgs.append(ImageStack(fov_stack, fov=fov))

    return imgs


def load_init_scaling_factors(imgs):
    init_scaling_factors = []

    for i in range(imgs[0].num_frames):
        X = []

        for x in imgs:
            c = np.percentile(x.imgs[i], 99.9)

            X.append(x.imgs[i][x.imgs[i] > c])

        X = np.concat(X)

        init_scaling_factors.append(np.median(X))

    return np.array(init_scaling_factors)


def write_results(codebook_file, file_name, imgs, pipeline):
    codebook = load_codebook(codebook_file, drop_blanks=False)

    pipeline.decoder.decoder = NearestNeigbourPixelDecoder(codebook, max_dist=0.3321, min_norm=0.4)

    df = []

    for fov, result in enumerate(pipeline.predict(imgs)):
        s = result.spots

        s = s.assign(fov=fov)

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
