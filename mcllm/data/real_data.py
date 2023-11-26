from typing import List
import pandas as pd
import torch.utils.data as data
import numpy as np
import torch
from mcllm.data.util import get_register_mask, get_att_mask, get_nan_mask, get_masks
from os.path import dirname, join
from mcllm.config import path_to_repo
import os


class RealDataDataset(data.Dataset):
    '''
    Dataset that returns real-data matrices

    Params
    ------
    static: bool
        Decides whether the same matrix should be returned at the same index every time
    '''

    def __init__(
        self,
        m_list: List[int],
        n_list: List[int],
        frac_nan_rand_mask_list: List[float],
        frac_nan_col_mask_list: List[float],
        frac_col_vs_random: float,
        use_rowcol_attn: bool = False,
        length=100, seed=13, randomize=False,
        n_registers=1,
    ):
        if isinstance(m_list, int):
            m_list = [m_list]
        if isinstance(n_list, int):
            n_list = [n_list]
        if isinstance(frac_nan_rand_mask_list, float):
            frac_nan_rand_mask_list = [frac_nan_rand_mask_list]
        self.m_list = m_list
        self.n_list = n_list
        self.m_max = max(m_list)
        self.n_max = max(n_list)
        self.frac_nan_rand_mask_list = frac_nan_rand_mask_list
        self.frac_nan_col_mask_list = frac_nan_col_mask_list
        self.frac_col_vs_random = frac_col_vs_random
        self.seed = seed
        self.length = length
        self.randomize = randomize
        self.use_rowcol_attn = use_rowcol_attn
        self.n_registers = n_registers
        self.register_mask = get_register_mask(
            self.m_max, self.n_max, n_registers)

        self.N_DSETS = 797
        self.dset_dir = join(path_to_repo, 'data',
                             'tabular-benchmark-797-classification')
        self.dset_list = [d for d in os.listdir(
            self.dset_dir) if d.endswith('.csv')]

    def __len__(self):
        return self.length

    def get_real_data(self, m, n):
        rng = self.rng
        dset_idx = rng.choice(self.N_DSETS)
        dset_name = self.dset_list[dset_idx]
        # mat = np.loadtxt(join(self.dset_dir, dset_name), delimiter=',')
        mat = pd.read_csv(join(self.dset_dir, dset_name), delimiter=',').values

        # select a random subset of the rows and columns
        if m < mat.shape[0]:
            row_idx = rng.choice(mat.shape[0], m, replace=False)
            mat = mat[row_idx, :]
        if n < mat.shape[1]:
            col_idx = rng.choice(mat.shape[1], n, replace=False)
            mat = mat[:, col_idx]

        return mat

    def __getitem__(self, idx):
        if self.randomize:
            self.rng = np.random.default_rng(seed=None)
        else:
            self.rng = np.random.default_rng(self.seed + idx)

        # sample matrix params
        m = self.rng.choice(self.m_list)
        n = self.rng.choice(self.n_list)

        # create matrix
        x = self.get_real_data(m, n)
        x = (x - x.mean(axis=1).reshape(-1, 1)) / x.std(axis=1).reshape(-1, 1)

        # put matrix into full matrix
        x_clean = np.zeros((self.m_max + self.n_registers,
                            self.n_max + self.n_registers))
        x_clean[:m, :n] = x
        x_clean = torch.Tensor(x_clean.flatten())

        # get masks and x_nan
        x_nan, nan_mask, att_mask = get_masks(
            x_clean, self.m_max, self.n_max, m, n, self.n_registers,
            self.frac_nan_rand_mask_list, self.frac_nan_col_mask_list,
            self.frac_col_vs_random, self.use_rowcol_attn, self.rng)

        return x_nan, x_clean, nan_mask, att_mask, self.register_mask


if __name__ == '__main__':
    # from the mcllm/data repo
    # git clone https://huggingface.co/datasets/csinva/tabular-benchmark-797-classification

    # create a dataloader
    m = 10
    n = 10
    frac_nan_mask = 0.1
    seed = 13
    batch_size = 1
    dataset = RealDataDataset(
        [m], [n], [frac_nan_mask], [0], [0], seed, n_registers=0)
    print(dataset[0])
