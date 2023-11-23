import os.path
from os.path import join as oj
import pandas as pd
import requests
from typing import List
import torch.utils.data as data
import numpy as np
import torch
from mcllm.data.low_rank import get_register_mask, get_att_mask, get_nan_mask
from datasets import load_dataset


def download_dataset(dataset_name: str,
                      data_source:str = "tabular-benchmark" ):
    
    assert data_source in ["tabular-benchmark","imodels","sklearn"], "data source not recognized"

    if data_source == "tabular-benchmark":
        if not dataset_name.endswith("csv"):
                dataset_name = dataset_name + ".csv"
        dataset = pd.DataFrame(load_dataset('imodels/tabular-benchmark-797-classification',data_files = dataset_name)['train'])
        if 'Unnamed: 0.1' in dataset.columns:
            dataset = dataset.drop('Unnamed: 0.1',axis = 1)
        if 'Unnamed: 0' in dataset.columns:
            dataset = dataset.drop('Unnamed: 0',axis = 1)        
    
    elif data_source == "imodels":
        raise ValueError("Not implemented yet")
    elif data_source == "sklearn":
        raise ValueError("Not implemented yet")
    else:
        raise ValueError("Data source not recognized yet")

    return dataset



class RealDataset(data.Dataset):
    
    #Create missing data from real matrices
    
    def __init__(self,
                 dataset_name: str, #name of dataset
                 m_list: List[int], #number of rows to subsample 
                 n_list: List[int], #number of columns to subsample 
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
        if isinstance(frac_nan_rand_mask_list, float):
            frac_nan_rand_mask_list = [frac_nan_rand_mask_list]
        if isinstance(frac_nan_col_mask_list,float):
            frac_nan_col_mask_list = [frac_nan_col_mask_list]
        self.dataset_name = dataset_name
        self.m_list = m_list
        self.n_list = n_list
        self.m_max = max(m_list)
        self.n_max = max(n_list)
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

    def get_dataset(self,m,n):
        df = download_dataset(self.dataset_name)
        # Randomly sample m rows
        sampled_rows = df.sample(n=m)
        sampled_columns = np.random.choice(df.columns, size=n, replace=False)
        sampled_df = sampled_rows[sampled_columns]
        return sampled_df.values, sampled_df.columns.tolist()
    
    def __getitem__(self, idx):
        if self.randomize:
            self.rng = np.random.default_rng(seed=None)
        else:
            self.rng = np.random.default_rng(self.seed + idx)

        # sample matrix params
        m = self.rng.choice(self.m_list)
        n = self.rng.choice(self.n_list)
       
        # create matrix
        x,col_names = self.get_dataset(m, n)
       

        # put matrix into full matrix
        x_clean = np.zeros((self.m_max + self.n_registers,
                            self.n_max + self.n_registers))
        x_clean[:m, :n] = x
        x_clean = torch.Tensor(x_clean.flatten())

        # get masks and x_nan
        nan_mask,use_col_mask,frac_missing = get_nan_mask(
            self.m_max, self.n_max, self.n_registers, m, n,
            self.frac_nan_rand_mask_list, self.frac_nan_col_mask_list, self.frac_col_vs_random,
            rng=self.rng
        )
        x_nan = x_clean.clone()
        x_nan[nan_mask.bool()] = 0

        att_mask = get_att_mask(
            self.m_max, self.n_max, self.n_registers, m, n, self.use_rowcol_attn)

        return x_nan, x_clean, nan_mask, att_mask, self.register_mask,self.n_registers,x, use_col_mask,frac_missing,self.m_max,self.n_max


if __name__ == '__main__':
    m = 10
    n = 10
    frac_nan_mask = 0.1
    seed = 13
    batch_size = 1
    dataset = RealDataset('2dplanes',m, n, frac_nan_mask, seed)
    