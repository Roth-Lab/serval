import pathlib
import re

import pandas as pd
import numpy as np
import skimage

from serval.codebook import Codebook


class MerlinDataOrganisation(object):
    def __init__(
        self,
        codebook,
        file_name,
        img_dir,
        flip_horizontal=False,
        flip_vertical=False,
        transpose=False,
    ):
        self._init_df(codebook, file_name)

        self.img_dir = pathlib.Path(img_dir)

        self.flip_horizontal = flip_horizontal

        self.flip_vertical = flip_vertical

        self.transpose = transpose

        self._init_img_paths()

    @property
    def bit_to_color_map(self):
        """Zero indexed mapping of bit to color"""
        return self.df.set_index("bit_idx")["color"].to_dict()

    @property
    def bit_to_round_map(self):
        """Zero indexed mapping of bit to imaging round"""
        return self.df.set_index("bit_idx")["img_idx"].to_dict()

    @property
    def fovs(self):
        return sorted(set([x[0] for x in self._img_paths]))

    @property
    def img_rounds(self):
        return sorted(set([x[1] for x in self._img_paths]))

    @property
    def num_z_slices(self):
        return len(self.df["zPos"].iloc[0])

    def get_fiducial_img(self, fov, img_round):
        """Get image for fov and zero-based imaging round"""
        file_name = self._img_paths[(fov, img_round)]

        return self._load_image(file_name, self._get_fiducial_frame(img_round))

    def get_primary_img(self, bit, fov, z):
        """Get a primary image given the zero-based bit and z indices."""
        color = self._get_bit_color(bit)

        img_round = self._get_bit_img_round(bit)

        file_name = self._img_paths[(fov, img_round)]

        return self._load_image(file_name, self._get_primary_frame(color, img_round, z))

    def _get_bit_color(self, bit):
        """Get the color for zero based bit index"""
        return self.df.set_index("bit_idx").loc[bit, "color"]

    def _get_bit_img_round(self, bit):
        """Get zero indexed imaging round for zero based bit index"""
        return self.df.set_index("bit_idx").loc[bit, "img_idx"]

    def _get_fiducial_frame(self, img_round):
        """Get fiducial frame for zero based img_round"""
        return (
            self.df[["img_idx", "fiducialFrame"]]
            .drop_duplicates()
            .set_index("img_idx")
            .loc[img_round, "fiducialFrame"]
        )

    def _get_primary_frame(self, color, img_round, z):
        """Get primary frame for zero based imaging round and z index"""
        row = self.df.set_index(["img_idx", "color"]).loc[(img_round, color)]

        frame_idx = row["z_idx"][z]

        return row["frame"][frame_idx]

    def _init_df(self, codebook, file_name):
        df = pd.read_csv(file_name, converters={"bitNumber": int, "imagingRound": int})

        df = df[df["imagingRound"] != -1]

        # Sort df based on codebook order
        df = df.set_index("readoutName")
        df = df.loc[codebook.readout_names]
        df = df.reset_index()
        df = df.rename(columns={"index": "readoutName"})
        
        # Validate the bitNumbers are unique
        assert df["bitNumber"].nunique() == df.shape[0]

        # Reset bitNumber to match codebook order and zero index contiguous
        df["bit_idx"] = df.index

        # Map imaging rounds to zero index contiguous
        self._img_round_map = dict(
            zip(
                sorted(df["imagingRound"].unique()), range(df["imagingRound"].nunique())
            )
        )

        df["img_idx"] = df["imagingRound"].map(self._img_round_map)

        df["frame"] = df["frame"].apply(_parse_str_to_list)
        df["frame"] = df["frame"].apply(lambda x: [int(y) for y in x])

        df["zPos"] = df["zPos"].apply(_parse_str_to_list)

        self._zpos_map = dict(zip(df["zPos"].iloc[0], range(len(df["zPos"].iloc[0]))))

        df["z_idx"] = df["zPos"].apply(lambda x: [self._zpos_map[y] for y in x])

        self.df = df

    def _init_img_paths(self):
        p = re.compile(self.df["imageRegExp"].iloc[0])

        self._img_paths = {}

        for x in self.img_dir.glob("*"):
            m = p.search(str(x))

            if m is None:
                continue

            fov = int(m.groupdict()["fov"])

            img_round = int(m.groupdict()["imagingRound"])

            if img_round not in self.df["imagingRound"].unique():
                continue

            # Convert to zero based index
            img_round = self._img_round_map[img_round]

            self._img_paths[(fov, img_round)] = x

    def _load_image(self, file_name, frame):
        img = skimage.io.imread(file_name, key=frame)

        if self.transpose:
            img = img.T

        if self.flip_horizontal:
            img = np.flip(img, axis=1)

        if self.flip_horizontal:
            img = np.flip(img, axis=0)

        return img


class MerlinDataset(object):
    def __init__(
        self,
        codebook_file,
        data_org_file,
        img_dir,
        flip_horizontal=False,
        flip_vertical=False,
        transpose=False,
    ):
        self.codebook = self._load_codebook(codebook_file)

        self.data_org = MerlinDataOrganisation(
            self.codebook,
            data_org_file,
            img_dir,
            flip_horizontal=flip_horizontal,
            flip_vertical=flip_vertical,
            transpose=transpose,
        )

    def get_fiducial_image_stack(self, fov, crop_size=0):
        imgs = []

        for r in self.data_org.img_rounds:
            img = self.data_org.get_fiducial_img(fov, r)

            if crop_size > 0:
                img = skimage.util.crop(img, crop_size)

            imgs.append(img)

        return np.array(imgs)

    def get_primary_image_stack(self, fov, z, crop_size=0):
        imgs = []

        for bit in range(self.codebook.num_bits):
            img = self.data_org.get_primary_img(bit, fov, z)

            if crop_size > 0:
                img = skimage.util.crop(img, crop_size)

            imgs.append(img)

        return np.array(imgs)

    def _load_codebook(self, file_name):
        df = pd.read_csv(file_name)

        df = df.drop("id", axis=1)

        df = df.set_index("name")

        return Codebook(df)


def _parse_str_to_list(frame_str):
    frame_list = []
    for x in frame_str.replace("[", "").replace("]", "").split(" "):
        if len(x) == 0:
            continue

        frame_list.append(x)

    return frame_list
