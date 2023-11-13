from typing import List
import torch.utils.data as data
import numpy as np
import torch


from mcllm.data.synthetic import LowRankDataset


DATALOADERS = [
    ('low_rank_m10_n10_r3_fracnan01', data.DataLoader(LowRankDataset(10, 10, 3, frac_nan_rand_mask_list = 0.1,frac_nan_col_mask_list = 0.1),
                                                      batch_size=1, shuffle=True))]
    
