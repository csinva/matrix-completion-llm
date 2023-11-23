from typing import List
import torch.utils.data as data
import numpy as np
import torch


from mcllm.data import *


DATALOADERS = [('low_rank', LowRankDataset(10, 10, 3, frac_nan_rand_mask_list = 0.1,frac_nan_col_mask_list = 0.1,frac_col_vs_random = 0.5))]
    
