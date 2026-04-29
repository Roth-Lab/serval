"""
Module for organizing MERlin image datasets and loading image stacks.
"""

import pathlib
import re

import pandas as pd
import numpy as np
import skimage

from serval.codebook import Codebook


def _parse_str_to_list(frame_str):
    """Parse a string representation of a list of frame indices into a Python list.

    Args:
        frame_str (str): A string like "[0 1 2]" or "[3 4]".

    Returns:
        list of str: The list of frame tokens extracted from the string.
    """
    frame_list = []
    for x in frame_str.replace("[", "").replace("]", "").split(" "):
        if len(x) == 0:
            continue

        frame_list.append(x)

    return frame_list


class MerlinDataOrganisation(object):
    """Load and organize MERlin dataset metadata for image file lookup.

    This class reads a CSV describing bit-to-round mappings and image frames,
    then indexes on field-of-view (FOV) and imaging round to locate files
    and specific frames.

    Attributes:
        img_dir (Path): Directory containing raw image files.
        flip_horizontal (bool): Whether to flip images horizontally.
        flip_vertical (bool): Whether to flip images vertically.
        transpose (bool): Whether to transpose images.
    """

    def __init__(
        self,
        codebook,
        file_name,
        img_dir,
        flip_horizontal=False,
        flip_vertical=False,
        transpose=False,
    ):
        """Initialize a MerlinDataOrganisation.

        Args:
            codebook (Codebook): Codebook instance with readout names.
            file_name (str or Path): Path to the CSV describing bit-round mappings.
            img_dir (str or Path): Directory containing raw image files.
            flip_horizontal (bool): If True, flip images horizontally.
            flip_vertical (bool): If True, flip images vertically.
            transpose (bool): If True, transpose images.
        """
        self._init_df(codebook, file_name)

        self.img_dir = pathlib.Path(img_dir)

        self.flip_horizontal = flip_horizontal

        self.flip_vertical = flip_vertical

        self.transpose = transpose

        self._init_img_paths()

    @property
    def bit_to_color_map(self):
        """
        dict: Zero indexed mapping of bit to color.
        """
        return self.df.set_index("bit_idx")["color"].to_dict()

    @property
    def bit_to_round_map(self):
        """
        dict: Zero indexed mapping of bit to imaging round.
        """
        return self.df.set_index("bit_idx")["img_idx"].to_dict()

    @property
    def fovs(self):
        """
        list of int: Sorted list of available field-of-view indices.
        """
        return sorted(set([x[0] for x in self._img_paths]))

    @property
    def img_rounds(self):
        """
        list of int: Sorted list of zero-based imaging round indices.
        """
        return sorted(set([x[1] for x in self._img_paths]))

    @property
    def num_z_slices(self):
        """
        int: Number of z-slices (from metadata).
        """
        return len(self.df["zPos"].iloc[0])

    def get_fiducial_img(self, fov, img_round):
        """
        Load the fiducial image for a given FOV and zero-based imaging round.

        Args:
            fov (int): Field-of-view index.
            img_round (int): Zero-based imaging round index.

        Returns:
            np.ndarray: The image frame for the fiducial channel.
        """
        file_name = self._img_paths[(fov, img_round)]

        return self._load_image(file_name, self._get_fiducial_frame(img_round))

    def get_primary_img(self, bit, fov, z):
        """
        Load a primary image for a given zero-bsaed bit, FOV, and z-slice.

        Args:
            bit (int): Zero-based bit index.
            fov (int): Field-of-view index.
            z (int): Zero-based z-slice index.

        Returns:
            np.ndarray: The image frame for the specified bit and z.
        """
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
            self.df[["img_idx", "fiducialFrame"]].drop_duplicates().set_index("img_idx").loc[img_round, "fiducialFrame"]
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
        self._img_round_map = dict(zip(sorted(df["imagingRound"].unique()), range(df["imagingRound"].nunique())))

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
        """
        Read and optionally transpose/flip an image frame from disk.

        Args:
            file_name (Path): Path to the image file.
            frame (int): Frame index to load via skimage.

        Returns:
            np.ndarray: The loaded (and transformed) image.
        """
        img = skimage.io.imread(file_name, key=frame)

        if self.transpose:
            img = img.T

        if self.flip_horizontal:
            img = np.flip(img, axis=1)

        if self.flip_horizontal:
            img = np.flip(img, axis=0)

        return img


class MerlinDataset(object):
    """
    High-level interface to load MERlin codebook and image stacks.

    This wraps Codebook parsing and MerlinDataOrganisation to provide easy
    access to fiducial and primary image stacks.

    Attributes:
        codebook (Codebook): Parsed codebook instance.
        data_org (MerlinDataOrganisation): Metadata & image-path organizer.
    """

    def __init__(
        self,
        codebook_file,
        data_org_file,
        img_dir,
        flip_horizontal=False,
        flip_vertical=False,
        transpose=False,
    ):
        """
        Initialize a MerlinDataset instance.

        Args:
            codebook_file (str): Path to the codebook CSV.
            data_org_file (str): Path to the data organization CSV.
            img_dir (str): Directory containing raw image files.
            flip_horizontal (bool): Flip images horizontally if True.
            flip_vertical (bool): Flip images vertically if True.
            transpose (bool): Transpose images if True.
        """
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
        """
        Load a stack of fiducial images across all rounds for a given FOV.

        Args:
            fov (int): Field-of-view index.
            crop_size (int): Number of pixels to crop from each edge. Defaults to 0.

        Returns:
            np.ndarray: Array of shape (rounds, height, width).
        """
        imgs = []

        for r in self.data_org.img_rounds:
            img = self.data_org.get_fiducial_img(fov, r)

            if crop_size > 0:
                img = skimage.util.crop(img, crop_size)

            imgs.append(img)

        return np.array(imgs)

    def get_primary_image_stack(self, fov, z, crop_size=0):
        """
        Load a stack of primary images for all bits at a given FOV and z-slice.

        Args:
            fov (int): Field-of-view index.
            z (int): Zero-based z-slice index.
            crop_size (int): Pixels to crop from each edge. Defaults to 0.

        Returns:
            np.ndarray: Array of shape (bits, height, width).
        """
        imgs = []

        for bit in range(self.codebook.num_bits):
            img = self.data_org.get_primary_img(bit, fov, z)

            if crop_size > 0:
                img = skimage.util.crop(img, crop_size)

            imgs.append(img)

        return np.array(imgs)

    def _load_codebook(self, file_name):
        """
        Load a Codebook from CSV, dropping any unwanted columns.

        Args:
            file_name (str): Path to the codebook CSV with columns ['name', ...].

        Returns:
            Codebook: Parsed codebook instance.
        """
        df = pd.read_csv(file_name)

        df = df.drop("id", axis=1)

        df = df.set_index("name")

        return Codebook(df)
