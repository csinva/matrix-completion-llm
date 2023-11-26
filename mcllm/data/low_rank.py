from typing import List
import torch.utils.data as data
import numpy as np
import torch
from mcllm.data.util import get_register_mask, get_att_mask, get_nan_mask, get_masks


class LowRankDataset(data.Dataset):
    '''
    Dataset that returns low-rank matrices

    Params
    ------
    static: bool
        Decides whether the same matrix should be returned at the same index every time
    '''

    def __init__(
        self,
        m_list: List[int],
        n_list: List[int],
        rank_list: List[int],
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
        if isinstance(rank_list, int):
            rank_list = [rank_list]
        if isinstance(frac_nan_rand_mask_list, float):
            frac_nan_rand_mask_list = [frac_nan_rand_mask_list]
        self.m_list = m_list
        self.n_list = n_list
        self.m_max = max(m_list)
        self.n_max = max(n_list)
        self.rank_list = rank_list
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

    def __len__(self):
        return self.length

    def get_low_rank_matrix(self, m, n, rank):
        rng = self.rng
        # A = rng.normal(size=(m, rank)) @ rng.normal(size=(rank, n))
        A = rng.uniform(size=(m, rank)) @ rng.uniform(size=(rank, n))
        return A

    def __getitem__(self, idx):
        if self.randomize:
            self.rng = np.random.default_rng(seed=None)
        else:
            self.rng = np.random.default_rng(self.seed + idx)

        # sample matrix params
        m = self.rng.choice(self.m_list)
        n = self.rng.choice(self.n_list)
        rank_list_filt = [r for r in self.rank_list if r < min(m, n) - 1]
        rank = self.rng.choice(rank_list_filt)

        # create matrix
        x = self.get_low_rank_matrix(m, n, rank)
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
    # create a dataloader
    m = 10
    n = 10
    rank = 3
    frac_nan_mask = 0.1
    seed = 13
    batch_size = 1
    dataset = LowRankDataset(m, n, rank, frac_nan_mask, seed)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
