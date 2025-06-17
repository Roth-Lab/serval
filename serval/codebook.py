import numpy as np
import pandas as pd


class Codebook(object):
    def __init__(self, df):
        self.df = df

    @staticmethod
    def from_file(file_name, add_background=False, blank_prefix=None):
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
        return self.df.values

    @property
    def num_bits(self):
        return self.df.shape[1]

    @property
    def num_targets(self):
        return self.df.shape[0]

    @property
    def readout_names(self):
        return self.df.columns

    @property
    def targets(self):
        return self.df.index

    def get_barcode(self, target):
        return self.df.loc[target]

    def get_readout_name(self, bit_index):
        return self.df.columns[bit_index]

    def get_target_id(self, target):
        return list(self.df.index).index(target)
