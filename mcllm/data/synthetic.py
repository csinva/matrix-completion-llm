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

    def __init__(self, m, n, rank, frac_nan_mask, length=100, seed=13, randomize=False):
        self.m = m
        self.n = n
        self.rank = rank
        self.frac_nan_mask = frac_nan_mask
        self.seed = seed
        self.length = length
        self.randomize = randomize

    def __len__(self):
        return self.length

    def create_low_rank_matrix(self, m, n, rank, seed=13, idx=None):
        rng = self.rng
        # A = rng.normal(size=(m, rank)) @ rng.normal(size=(rank, n))
        A = rng.uniform(size=(m, rank)) @ rng.uniform(size=(rank, n))
        return A

    def __getitem__(self, idx):
        if self.randomize:
            self.rng = np.random.default_rng(seed=None)
        else:
            self.rng = np.random.default_rng(self.seed + idx)
        x = self.create_low_rank_matrix(
            self.m, self.n, self.rank)
        x = (x - x.mean(axis=1).reshape(-1, 1)) / x.std(axis=1).reshape(-1, 1)
        x = x.flatten()

        nan_mask = self.rng.binomial(1, self.frac_nan_mask, size=x.shape)
        nan_mask_t = torch.Tensor(nan_mask)
        x_clean_t = torch.Tensor(x)
        x_nan_t = x_clean_t.clone()
        x_nan_t[nan_mask_t.bool()] = 0

        return x_nan_t, x_clean_t, nan_mask_t


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
