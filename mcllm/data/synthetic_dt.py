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
        self.threshold = torch.rand(1).item() if self.feature is not None else None

        # Update the feature range for this node
        self.feature_ranges = feature_ranges.clone()
        self.is_leaf = depth >= max_depth or not available_features
        self.data = []
        
        if not self.is_leaf:
            split_point = self.feature_ranges[self.feature][0] + self.threshold * (self.feature_ranges[self.feature][1] - self.feature_ranges[self.feature][0])
            self.feature_ranges[self.feature][1] = split_point

            left_ranges = self.feature_ranges.clone()
            left_ranges[self.feature][1] = split_point
            right_ranges = self.feature_ranges.clone()
            right_ranges[self.feature][0] = split_point

            remaining_features = [f for f in available_features if f != self.feature]
            self.left = RandomDecisionTreeNode(depth + 1, max_depth, n_features, left_ranges, remaining_features)
            self.right = RandomDecisionTreeNode(depth + 1, max_depth, n_features, right_ranges, remaining_features)
        else:
            self.left = None
            self.right = None
    
    def generate_vector(self,return_leaf = True):
        if self.is_leaf:
            vector = torch.rand(self.n_features).mul(self.feature_ranges[:, 1] - self.feature_ranges[:, 0]) + self.feature_ranges[:, 0]
            #np.array([np.random.uniform(low, high) for low, high in self.feature_ranges])
            self.data.append(vector)
            return (vector, self) if return_leaf else vector
        elif torch.rand(1).item() < self.threshold:
            return self.left.generate_vector(return_leaf)
        else:
            return self.right.generate_vector(return_leaf)
    
    def compute_leaf_averages(self,averages = None,leaves = None):
        
        if averages is None:
            averages = []
        if leaves is None:
            leaves = []
        
        if self.is_leaf:
            if self.data:
                # Compute the average for the current leaf node
                leaf_data = torch.stack(self.data)
                leaf_average = leaf_data.mean(dim=0)  # Average across all dimensions for each vector
                averages.append(leaf_average)
                leaves.append(self)
        else:
            if self.left:
                self.left.compute_leaf_averages(averages, leaves)
            if self.right:
                self.right.compute_leaf_averages(averages, leaves)

        return averages,leaves





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
                append_label: bool = True,
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
        self.append_label = append_label
        if self.append_label: 
            n_list = [n+1 for n in n_list]
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
        initial_ranges = torch.tensor([[0, 1]] * n, dtype=torch.float32)
        available_features = list(range(n))
        root = RandomDecisionTreeNode(0, depth, n, initial_ranges, available_features)
        vectors = []
        leaf_nodes = []
        for _ in range(m):
            vector,leaf = root.generate_vector(return_leaf = True)
            vectors.append(vector)
            leaf_nodes.append(leaf)
        
        # Convert the list of vectors to a PyTorch tensor
        tensor_vectors = torch.stack(vectors)

        # Compute the averages of each leaf
        leaf_averages,leaves = root.compute_leaf_averages()
        
        # Create a mapping from leaf node to its average
        leaf_to_average = {leaf: avg for leaf, avg in zip(leaves, leaf_averages)}

        averages_for_each_vector = torch.tensor([leaf_to_average[leaf].mean().item() for leaf in leaf_nodes])

        # Concatenate this tensor of averages with the original tensor_vectors
        tensor_vectors_with_averages = torch.cat((tensor_vectors, averages_for_each_vector.unsqueeze(1)), dim=1)


        #tensor_vectors_with_averages = torch.stack(augmented_vectors)

        return tensor_vectors_with_averages
    
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
        if self.append_label:
            x = self.get_DT_matrix(m, n - 1, depth)
        else:
            x = self.get_DT_matrix(m, n , depth)
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

    

        