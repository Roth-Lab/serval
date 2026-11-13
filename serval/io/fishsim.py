"""
Module for organizing FishSim-generated image datasets and loading image stacks.
This module supports loading a FishSim image folder that contains multiple FOV files.
For example, {XXX_000.ext, XXX_001.ext, XXX_002.ext}, where ext can be tif,tiff, ignore case.
"""
import pathlib
import re

import pandas as pd
import numpy as np
import skimage

from serval.codebook import Codebook

class FishsimDataOrganisation(object):
    
    def __init__(
        self,
        codebook,
        img_dir,
    ):
        
        #self._init_df(codebook, file_name)

        self.img_dir = pathlib.Path(img_dir)

        self._init_img_paths()
        
    @property
    def fovs(self):
        """
        list of int: Sorted list of available field-of-view indices.
        """
        return sorted(self._img_paths.keys())
        
    def _init_img_paths(self):
        # We are defining fov as a key that can be loaded from RegExpr (re) search results
        p = re.compile("_(?P<fov>\d+)\.tiff?$")

        self._img_paths = {}

        for x in self.img_dir.glob("*"):
            m = p.search(str(x))

            if m is None:
                continue

            fov = int(m.groupdict()["fov"])

            if m is None or "fov" not in m.groupdict():
                continue

            self._img_paths[fov] = x
            
            
    def get_primary_img(self, fov):
        """
        Load a primary image for a given FOV.

        Args:
            fov (int): Field-of-view index.

        Returns:
            np.ndarray: The image frame for the specified bit and z.
        """
        # Since we only have 1 dict key, _img_paths[fov] same as _img_paths[(fov)]
        file_name = self._img_paths[fov]

        return self._load_image(file_name)
    
    
    def _load_image(self, file_name):
        """
        Load the entire Fishsim-generated N-bit stacked image.
        This is different from merlin.py which loads image by frame based on data org.

        Args:
            file_name (Path): Path to the image file.

        Returns:
            np.ndarray: The loaded (and transformed) image.
        """
        img = skimage.io.imread(file_name)

        return img
            
        
class FishsimDataset(object):
    """
    High-level interface to load MERlin codebook and image stacks.

    This wraps Codebook parsing and MerlinDataOrganisation to provide easy
    access to fiducial and primary image stacks.

    Attributes:
        codebook (Codebook): Parsed codebook instance.
    """

    def __init__(
        self,
        codebook_file,
        img_dir,
    ):
        """
        Initialize a MerlinDataset instance.

        Args:
            codebook_file (str): Path to the codebook CSV.
            img_dir (str): Directory containing raw image files.
        """
        self.codebook = self._load_codebook(codebook_file)
        
        self.data_org = FishsimDataOrganisation(
            self.codebook,
            img_dir,
        )


    def get_primary_image_stack(self, fov):
        """
        Load a stack of primary images for all bits.

        Args:
            fov (int): Field-of-view index.

        Returns:
            np.ndarray: Array of shape (bits, height, width).
        """
        return np.array(self.data_org.get_primary_img(fov))


    def _load_codebook(self, file_name):
        """
        Load a Codebook from CSV, dropping any unwanted columns.

        Args:
            file_name (str): Path to the codebook CSV with columns ['name', ...].

        Returns:
            Codebook: Parsed codebook instance.
        """
        # Enforce CSV only codebook file
        if pathlib.Path(file_name).suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, but got: {file_name}")
    
        df = pd.read_csv(file_name)

        df = df.drop("id", axis=1, errors="ignore")

        df = df.set_index("name" if "name" in df.columns else "target")

        return Codebook(df)
