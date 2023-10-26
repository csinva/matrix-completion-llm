from typing import List
import torch.utils.data as data
import numpy as np
import torch


# write a pytorch
class LowRankDataset(data.Dataset):
    '''
    Dataset that returns low-rank matrices

    Params
    ------
    static: bool
        Decides whether the same matrix should be returned at the same index every time
    '''

    def __init__(self, m_list: List[int], n_list: List[int], rank_list: List[int],
                 frac_nan_mask_list: List[float], length=100, seed=13, randomize=False):
        self.m_list = m_list
        self.n_list = n_list
        self.m_max = max(m_list)
        self.n_max = max(n_list)
        self.rank_list = rank_list
        self.frac_nan_mask_list = frac_nan_mask_list
        self.seed = seed
        self.length = length
        self.randomize = randomize

    def __len__(self):
        return self.length

    def create_low_rank_matrix(self, m, n, rank):
        rng = self.rng
        # A = rng.normal(size=(m, rank)) @ rng.normal(size=(rank, n))
        A = rng.uniform(size=(m, rank)) @ rng.uniform(size=(rank, n))
        return A

    def __getitem__(self, idx):
        if self.randomize:
            self.rng = np.random.default_rng(seed=None)
        else:
            self.rng = np.random.default_rng(self.seed + idx)

        x_full = np.zeros((self.m_max, self.n_max))

        # sample matrix params
        m = self.rng.choice(self.m_list)
        n = self.rng.choice(self.n_list)
        rank_list_filt = [r for r in self.rank_list if r < min(m, n) - 1]
        rank = self.rng.choice(rank_list_filt)
        frac_nan_mask = self.rng.choice(self.frac_nan_mask_list)

        # create matrix
        x = self.create_low_rank_matrix(m, n, rank)
        x = (x - x.mean(axis=1).reshape(-1, 1)) / x.std(axis=1).reshape(-1, 1)
        x_full[:m, :n] = x

        # attention mask
        att_mask = np.zeros_like(x_full)
        att_mask[:m, :n] = 1
        att_mask = att_mask.flatten()

        # nan mask - randomly mask some frac
        # only mask values in first m rows and first n cols
        nan_mask_mini = self.rng.binomial(1, frac_nan_mask, size=(m, n))
        nan_mask = np.zeros_like(x_full)
        nan_mask[:m, :n] = nan_mask_mini
        nan_mask_t = torch.Tensor(nan_mask).flatten()

        x_full = x_full.flatten()
        x_clean_t = torch.Tensor(x_full)
        x_nan_t = x_clean_t.clone()
        x_nan_t[nan_mask_t.bool()] = 0
        att_mask_t = torch.Tensor(att_mask)
        # print('shapes data', x_nan_t.shape, att_mask_t.shape)

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
