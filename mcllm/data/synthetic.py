from typing import List
import torch.utils.data as data
import numpy as np
import torch


def get_register_mask(m_max, n_max, n_registers):
    '''Returns flattened mask that is 0 everywhere except for the registers.
    Note: registers are stored at the right / bottom of the sequence
    '''
    register_mask = torch.zeros((m_max + n_registers,
                                 n_max + n_registers))
    if n_registers > 0:
        register_mask[-n_registers:] = 1
        register_mask[:, -n_registers:] = 1
    return register_mask.flatten()


def get_att_mask(m_max, n_max, n_registers, m, n, use_rowcol_attn: bool):
    '''Create attention mask of size (seq_len, seq_len)
    Attention is set to 0 except for registers and points in the same row/col (up to m, n)
    Note: registers are stored at the right / bottom of the sequence
        Each row/col can attend to its own registers, and registers can attend to each other
        in addition to their own row/col
    '''
    seq_len = (m_max + n_registers) * \
        (n_max + n_registers)
    m_max_with_reg = m_max + n_registers

    # basic attn_mask
    att_mask_kernel = torch.zeros((seq_len, seq_len))
    # seq_len_before_reg = (m_max + n_registers) * n_max
    # everything attends to registers (also registers attend to each other)
    # att_mask_kernel[seq_len_before_reg:] = 1
    # att_mask_kernel[:, seq_len_before_reg:] = 1

    if use_rowcol_attn:
        for i in range(seq_len):
            r_idx_row = i // m_max_with_reg
            c_idx_row = i % m_max_with_reg
            for j in range(seq_len):
                r_idx_col = j // m_max_with_reg
                c_idx_col = j % m_max_with_reg

                # everything attends to points in the same row/col
                if r_idx_row == r_idx_col or c_idx_row == c_idx_col:
                    att_mask_kernel[i, j] = 1

                # register attention
                # if c_idx_col >= n_max:
                    # att_mask_kernel[i, j] = 1
    else:
        # attention mask for full attention
        att_mask_kernel[:m*n, :m*n] = 1
    return att_mask_kernel


def get_nan_mask(
    m_max, n_max, n_registers, m, n,
    frac_nan_rand_mask_list, frac_nan_col_mask_list, frac_col_vs_random,
    rng
):
    '''
    nan mask - randomly mask some frac (1 means nan)
    only mask values in first m rows and first n cols
    '''
    nan_mask = np.zeros((m_max + n_registers,
                        n_max + n_registers))
    use_col_mask = rng.binomial(1, frac_col_vs_random)
    if use_col_mask:
        # set fraction of a single random column to nan
        frac_nan_col_mask = rng.choice(frac_nan_col_mask_list)
        col_idx = rng.choice(n)
        nan_mask[:m, col_idx] = rng.binomial(
            1, frac_nan_col_mask, size=(m, 1)).flatten()
    else:
        frac_nan_rand_mask = rng.choice(frac_nan_rand_mask_list)
        nan_mask[:m, :n] = rng.binomial(1, frac_nan_rand_mask, size=(m, n))
    return torch.Tensor(nan_mask).flatten()


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
        nan_mask = get_nan_mask(
            self.m_max, self.n_max, self.n_registers, m, n,
            self.frac_nan_rand_mask_list, self.frac_nan_col_mask_list, self.frac_col_vs_random,
            rng=self.rng
        )
        x_nan = x_clean.clone()
        x_nan[nan_mask.bool()] = 0

        att_mask = get_att_mask(
            self.m_max, self.n_max, self.n_registers, m, n, self.use_rowcol_attn)

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
