from typing import List
import torch.utils.data as data
import numpy as np
import torch
import random
from synthetic import get_register_mask, get_att_mask, get_nan_mask

class RandomDecisionTreeNode:
    def __init__(self,depth,max_depth,n_features,feature_ranges,available_features):
        self.depth = depth
        self.n_features = n_features
        self.feature = random.choice(available_features) if available_features else None
        self.threshold = random.random() if self.feature is not None else None

        # Update the feature range for this node
        self.feature_ranges = feature_ranges.copy()
        self.is_leaf = depth >= max_depth or not available_features

        if not self.is_leaf:
            split_point = self.feature_ranges[self.feature][0] 
            + self.threshold * (self.feature_ranges[self.feature][1] - self.feature_ranges[self.feature][0])
            self.feature_ranges[self.feature][1] = split_point

            left_ranges = self.feature_ranges.copy()
            left_ranges[self.feature][1] = split_point
            right_ranges = self.feature_ranges.copy()
            right_ranges[self.feature][0] = split_point

            remaining_features = [f for f in available_features if f != self.feature]
            self.left = RandomDecisionTreeNode(depth + 1, max_depth, n_features, left_ranges, remaining_features)
            self.right = RandomDecisionTreeNode(depth + 1, max_depth, n_features, right_ranges, remaining_features)
        else:
            self.left = None
            self.right = None
    
    def generate_vector(self):
        if self.is_leaf:
            return np.array([np.random.uniform(low, high) for low, high in self.feature_ranges])
        elif random.random() < self.threshold:
            return self.left.generate_vector()
        else:
            return self.right.generate_vector()






class DTDataset(data.Dataset):
    '''
    Dataset generated from decision tree
    '''
    def __init__(self,
                m_list: List[int], #number of rows
                n_list: List[int], #number of columns
                depth_list: List[int], #possible depths of decison trees
                frac_nan_rand_mask_list: List[float],
                frac_nan_col_mask_list: List[float],
                frac_col_vs_random: float = 0.0,
                use_rowcol_attn: bool = False,
                length=100, seed=13, randomize=False,
                n_registers=1,
                ):

        if isinstance(m_list, int):
            m_list = [m_list]
        if isinstance(n_list, int):
            n_list = [n_list]
        if isinstance(depth_list, int):
            depth_list = [depth_list]
        if isinstance(frac_nan_rand_mask_list, float):
            frac_nan_rand_mask_list = [frac_nan_rand_mask_list]
        self.m_list = m_list
        self.n_list = n_list
        self.m_max = max(m_list)
        self.n_max = max(n_list)
        self.depth_list = depth_list
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

    def get_DT_matrix(self, m, n, depth):
        initial_ranges = [[0, 1] for _ in range(n)] 
        available_features = list(range(n))
        root = RandomDecisionTreeNode(0, depth, n, initial_ranges, available_features)
        return torch.tensor([root.generate_vector() for _ in range(m)])

    def __getitem__(self, idx):
        if self.randomize:
            self.rng = np.random.default_rng(seed=None)
        else:
            self.rng = np.random.default_rng(self.seed + idx)

        # sample matrix params
        m = self.rng.choice(self.m_list)
        n = self.rng.choice(self.n_list)
        depth_list_filt = [d for d in self.depth_list if d < n - 1] 
        rank = self.rng.choice(depth_list_filt)

        # create matrix
        x = self.get_DT_matrix(m, n, depth)
        #x = (x - x.mean(axis=1).reshape(-1, 1)) / x.std(axis=1).reshape(-1, 1)

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
    depth = 3
    frac_nan_mask = 0.1
    seed = 13
    batch_size = 1
    dataset = DTDataset(m, n, depth, frac_nan_mask, seed)
    print(dataset[1][1])
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    

        