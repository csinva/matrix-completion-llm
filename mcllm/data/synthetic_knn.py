from typing import List
import torch.utils.data as data
import numpy as np
import torch
from synthetic import get_register_mask, get_att_mask, get_nan_mask



class KNNDataset(data.Dataset):
    '''
    Dataset generated from K centroids, where each centroid is separated by a distance eta, and 

    '''
    def __init__(self,
                 m_list: List[int], #number of rows
                 n_list: List[int], #number of columns
                 num_centroids_list: List[int], #number of centroids, 
                 distance_list: List[int], #distance of hyperspheres around each hypersphere,
                 frac_nan_rand_mask_list :  List[float], #fraction of entries to set to nan
                 frac_nan_col_mask_list: List[float], #fraction of a single column to nan
                 frac_col_vs_random: float = 0.0,
                 use_rowcol_attn: bool = False,
                 length=100, seed=13, randomize=False,
                  n_registers=1,):
        if isinstance(m_list, int):
            m_list = [m_list]
        if isinstance(n_list, int):
            n_list = [n_list]
        if isinstance(num_centroids_list, int):
            num_centroids_list = [num_centroids_list]
        if isinstance(distance_list,float) or isinstance(distance_list,double):
            distance_list = [distance_list]
        if isinstance(frac_nan_rand_mask_list, float):
            frac_nan_rand_mask_list = [frac_nan_rand_mask_list]
        self.m_list = m_list
        self.n_list = n_list
        self.m_max = max(m_list)
        self.n_max = max(n_list)
        self.num_centroids_list = num_centroids_list
        self.distance_list = distance_list
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
    
    def get_knn_matrix(self,m,n,num_centroids,distance):
        rng = self.rng
        centroids = torch.rand(num_centroids, n)
        vectors = []
        for _ in range(n):
            centroid = centroids[torch.randint(len(centroids), (1,))]
            new_vector = centroid + torch.randn(m) * distance
            vectors.append(new_vector)
        return torch.vstack(vectors)

    def __getitem__(self,idx):
        if self.randomize:
            self.rng = np.random.default_rng(seed=None)
        else:
            self.rng = np.random.default_rng(self.seed + idx)
        m = self.rng.choice(self.m_list)
        n = self.rng.choice(self.n_list)
        num_centroids = self.rng.choice(self.num_centroids_list)
        distance = self.rng.choice(self.distance_list)

        #create matrix
        x = self.get_knn_matrix(m, n, num_centroids,distance)

        #put matrix into full matrix
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
    num_centroids = 3
    distance = 0.1
    frac_nan_mask = 0.1
    seed = 13
    batch_size = 1
    dataset = KNNDataset(m, n, num_centroids, distance,frac_nan_mask, seed)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

