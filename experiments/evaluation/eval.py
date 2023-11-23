from typing import List
from collections import defaultdict
import argparse
import inspect
import warnings
import os
from os.path import join as oj
import pickle as pkl
import time
from tqdm import tqdm
import copy
import config 
from model_util import ModelConfig
import torch
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torch.nn.functional as F

from mcllm.model import *
from mcllm.data import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unsqueeze(tensor_tuple):
    return tuple(t.unsqueeze(0).to(device) for t in tensor_tuple)



def compare_estimators(estimators: List[ModelConfig],data,results_path):
    '''
    calculates results (RMSE) given estimators, dataset
    estimators: list of model configs
    dataset: Dataset object
    '''
    if type(estimators) != list:
        raise Exception("Needs to be list of matrix completion estimators")
    
    #store data type, m,n, masking mechanism, fraction_masked, RMSE , correlation between predictions 
    results = {}
    results['data_type'] = []
    results['num_rows'] = []
    results['num_cols'] = []
    results['masking_mechanism'] = []
    results['fraction_masked'] = []
    for est in estimators:
        results[est.name] = []
        results[est.name + "_predictions"] = []
    dset_name = data[0]
    dataset = data[1]
    for i in range(len(dataset)):
        x_nan, x_clean, nan_mask, att_mask,register_mask,n_registers,x,use_col_mask,frac_missing,m_max,n_max = dataset[i] #load matrix to complete 

        if use_col_mask == 1: #store masking mechanism 
            use_col_mask = 'col_mask'
        else:
            use_col_mask = 'MCAR'
        
        

        m,n = x.shape[0],x.shape[1]
        
        nan_mask_reshape = nan_mask.reshape(m_max + n_registers, n_max + n_registers)[:m,:n] #find indices of elements where they are masked 

        x_missing_value = copy.deepcopy(x)
        x_missing_value[nan_mask_reshape == 1] = np.nan #mask values

        x_target = x[nan_mask_reshape == 1]
        
        if len(x_target) == 0: #no missing values to impute 
            continue

        results['data_type'].append(dset_name)
        results['masking_mechanism'].append(use_col_mask)
        results['fraction_masked'].append(frac_missing)
        results['num_rows'].append(m)
        results['num_cols'].append(n)
        
        for est in estimators:
            if est.name == "McLLM":
                module = est.cls
                x_nan_un, x_clean_un, nan_mask_un, att_mask_un, register_mask_un = unsqueeze((x_nan, x_clean, nan_mask, att_mask, register_mask))
                output = module(x_nan_un, nan_mask_un, att_mask_un, register_mask_un, n_rows=m, n_columns=n)
                x_pred = output.detach().cpu().numpy().squeeze().reshape((m + n_registers, n + n_registers))[:m,:n]
                x_pred_at_nan = x_pred[nan_mask_reshape == 1]
                McLLM_error = np.mean((x_target - x_pred_at_nan) ** 2)
                results[est.name].append(McLLM_error)
                results[est.name + "_predictions"].append(x_pred_at_nan)
            else: #assume other methods have fit_transform method 
                model = est.cls()
                x_imputed = model.fit_transform(x_missing_value)
                x_imputed_at_nan = x_imputed[nan_mask_reshape == 1]
                results[est.name].append(np.mean((x_target - x_imputed_at_nan) ** 2))
                results[est.name + "_predictions"].append(x_imputed_at_nan)
        
    for est in estimators:
        print(est.name,results[est.name])
        print(" ")
        
    file_name = dset_name + ".pkl"
    results_file_name = os.path.join(args.results_path,file_name)
    with open(results_file_name, 'wb') as file:
        pkl.dump(results, file)
        


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, default = 'low_rank') 
    parser.add_argument('--downstream_task', type = str, default = 'matrix_completion') #todo: extend to downstream regression/classification
    parser.add_argument('--split_seed', type = int, default = 0) 
    parser.add_argument('--results_path', type=str,
                        default=oj(os.path.dirname(os.path.realpath(__file__)), 'evaluation_results'))

    args = parser.parse_args()
    DATALOADERS,ESTS = config.get_configs(args.config)    
  
    
    if len(ESTS) == 0:
        raise Exception("No estimators provided")
    if len(DATALOADERS) == 0:
        raise ValueError('No dataset provided')
  
    for data in tqdm(DATALOADERS):
        compare_estimators(ESTS,data,args.results_path)
      

    print('completed all experiments successfully!')



