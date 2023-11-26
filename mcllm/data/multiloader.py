import random
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import torch.utils.data as data
from mcllm.data.knn import KNNDataset
from mcllm.data.low_rank import LowRankDataset
import numpy.linalg as npl


class RandomConcatDataset(Dataset):
    def __init__(self, *datasets, mix_fracs=None):
        self.datasets = datasets
        if mix_fracs is None:
            mix_fracs = [1 / len(datasets)] * len(datasets)
        self.mix_fracs = mix_fracs
        self.lengths = [len(d) for d in datasets]
        self.total_length = sum(self.lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        dataset_idx = random.choices(
            range(len(self.datasets)), weights=self.mix_fracs)[0]
        dataset = self.datasets[dataset_idx]
        return dataset[index % len(dataset)]


if __name__ == '__main__':
    # create a dataloader
    m = 10
    n = 10
    num_centroids = 3
    distance = 0.1
    frac_nan_mask = 0.1
    seed = 13
    rank = 3
    batch_size = 1
    n_registers = 0
    dataset1 = KNNDataset(m, n, num_centroids, distance,
                          frac_nan_mask, seed, n_registers=n_registers)
    dataset2 = LowRankDataset(
        [m], [n], [rank], [frac_nan_mask], [0], [0], seed, n_registers=n_registers)

    random_concat_dataset = RandomConcatDataset(dataset1, dataset2)
    for i in range(10):
        x_nan, x_clean, nan_mask, att_mask, register_mask = random_concat_dataset[i]
        print('rank', npl.matrix_rank(x_clean.reshape(m, n)))
