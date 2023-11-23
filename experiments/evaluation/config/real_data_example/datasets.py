from typing import List
import torch.utils.data as data
import numpy as np
import torch


from mcllm.data.real_data import RealDataset


DATALOADERS = [('2dplanes', RealDataset('2dplanes',10, 10, frac_nan_rand_mask_list = 0.1,frac_nan_col_mask_list = 0.1,frac_col_vs_random = 0.5,length = 2))]
    
