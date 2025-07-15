import numpy as np
import pandas as pd


class Codebook(object):
    """A lookup table of barcode vectors for each target.

    Attributes:
        df (pd.DataFrame): Rows are targets, columns are readout bits.
    """

    def __init__(self, df):
        """Initialize Codebook with an existing DataFrame.

        Args:
            df (pd.DataFrame): DataFrame indexed by 'target' with bit columns.
        """
        self.df = df

    @staticmethod
    def from_file(file_name, add_background=False, blank_prefix=None):
        """Load a codebook from a TSV file and return a Codebook instance.

        Args:
            file_name (str): Path to a tab‑separated file with 'target' column.
            add_background (bool): If True, append a zero vector labeled 'background'.
            blank_prefix (str, optional): If provided, drop any targets whose
                names start with this prefix.
        
        Returns:
            Codebook: A new instance containing the loaded (and possibly
                filtered/augmented) barcode DataFrame.
        """
        df = pd.read_csv(file_name, index_col="target", sep="\t")

        if blank_prefix is not None:
            df = df.loc[~df.index.str.startswith(blank_prefix)]

        B = list(df.values)

        index = list(df.index)

        if add_background:
            B.append(np.zeros(len(B[0]), dtype=int))

            index.append("background")

        cb_df = pd.DataFrame(B, columns=df.columns, index=index)

        cb_df.index.name = "target"

        return Codebook(cb_df)

    @property
    def barcode_matrix(self):
        """np.ndarray: The raw barcode matrix (targets × bits)."""
        return self.df.values

    @property
    def num_bits(self):
        """int: Number of bits (columns) in each barcode."""
        return self.df.shape[1]

    @property
    def num_targets(self):
        """int: Number of targets (rows) in the codebook."""
        return self.df.shape[0]

    @property
    def readout_names(self):
        """Index: Column labels corresponding to readout bit names."""
        return self.df.columns

    @property
    def targets(self):
        """Index: Row labels corresponding to target names."""
        return self.df.index

    def get_barcode(self, target):
        """Retrieve the barcode vector for a given target.

        Args:
            target (str): The target name whose barcode to fetch.
        
        Returns:
            pd.Series: The bit vector (row) for the requested target.
        """
        return self.df.loc[target]

    def get_readout_name(self, bit_index):
        """Map a bit index to its readout column name.

        Args:
            bit_index (int): Zero‑based index into the barcode vector.
        
        Returns:
            str: The name of the readout corresponding to that bit.
        """
        return self.df.columns[bit_index]

    def get_target_id(self, target):
        """Convert a target name to its integer index.
        
        Args:
            target (str): The target name to look up.
        
        Returns:
            int: The zero‑based row index of the target in the DataFrame.
        """
        return list(self.df.index).index(target)
