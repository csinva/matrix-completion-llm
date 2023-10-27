from typing import List
import torch.utils.data as data
import numpy as np
import torch


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
        frac_nan_mask_list: List[float],
        use_rowcol_attn: bool = False,
        length=100, seed=13, randomize=False,
        n_registers=1,
    ):
        self.m_list = m_list
        self.n_list = n_list
        self.m_max = max(m_list)
        self.n_max = max(n_list)
        self.rank_list = rank_list
        self.frac_nan_mask_list = frac_nan_mask_list
        self.seed = seed
        self.length = length
        self.randomize = randomize
        self.use_rowcol_attn = use_rowcol_attn
        self.n_registers = n_registers

    def __len__(self):
        return self.length

    def create_low_rank_matrix(self, m, n, rank):
        rng = self.rng
        # A = rng.normal(size=(m, rank)) @ rng.normal(size=(rank, n))
        A = rng.uniform(size=(m, rank)) @ rng.uniform(size=(rank, n))
        return A

    def _create_att_mask(self, m, n):
        '''Create attention mask of size (seq_len, seq_len)
        Note: registers are stored at the right / bottom of the sequence
            Each row/col can attend to its own registers, and registers can attend to each other
            in addition to their own row/col
        '''
        seq_len = (self.m_max + self.n_registers) * \
            (self.n_max + self.n_registers)
        m_max_with_reg = self.m_max + self.n_registers
        if self.use_rowcol_attn:
            att_mask_kernel = np.zeros((seq_len, seq_len))
            for i in range(seq_len):
                r_idx_row = i // m_max_with_reg
                c_idx_row = i % m_max_with_reg
                for j in range(seq_len):
                    r_idx_col = j // m_max_with_reg
                    c_idx_col = j % m_max_with_reg

                    # everything attends to points in the same row/col
                    if r_idx_row == r_idx_col or c_idx_row == c_idx_col:
                        att_mask_kernel[i, j] = 1

                    # registers attend to each other
                    if r_idx_row >= self.n_max or r_idx_col >= self.n_max:
                        att_mask_kernel[i, j] = 1

        else:
            # attention mask for full attention (ignore registers for now)
            att_mask = np.zeros((self.m_max, self.n_max))
            att_mask[:m, :n] = 1
            att_mask = att_mask.flatten()
            # pytorch mha implementation only uses att_mask (size is just seq_len)
            att_mask_kernel = np.ones((seq_len, seq_len))
            att_mask_kernel[~att_mask.astype(bool)] = 0
            att_mask_kernel[:, ~att_mask.astype(bool)] = 0
        return torch.Tensor(att_mask_kernel)

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
        frac_nan_mask = self.rng.choice(self.frac_nan_mask_list)

        # create matrix
        x = self.create_low_rank_matrix(m, n, rank)
        x = (x - x.mean(axis=1).reshape(-1, 1)) / x.std(axis=1).reshape(-1, 1)

        # put matrix into full matrix
        x_full = np.zeros((self.m_max + self.n_registers,
                          self.n_max + self.n_registers))
        x_full[:m, :n] = x
        att_mask_t = self._create_att_mask(m, n)

        # nan mask - randomly mask some frac
        # only mask values in first m rows and first n cols
        nan_mask = np.zeros_like(x_full)
        nan_mask[:m, :n] = self.rng.binomial(1, frac_nan_mask, size=(m, n))
        nan_mask_t = torch.Tensor(nan_mask).flatten()

        x_full = x_full.flatten()
        x_clean_t = torch.Tensor(x_full)
        x_nan_t = x_clean_t.clone()
        x_nan_t[nan_mask_t.bool()] = 0

        return x_nan_t, x_clean_t, nan_mask_t, att_mask_t


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
