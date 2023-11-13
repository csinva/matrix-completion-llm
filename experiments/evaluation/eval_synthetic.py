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

import config 
from model_util import ModelConfig

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from mcllm.model import *
from mcllm.data import *



def compare_estimators(estimators: List[ModelConfig],
                       dataset):
    '''
    calculates results (RMSE) given estimators, synthetic dataset

    estimators: list of model configs
    dataset: Dataloader object
    '''
    if type(estimators) != list:
        raise Exception("Needs to be list of matrix completion estimators")

    results = defaultdict(lambda : [])
    
    

    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, default = 'low_rank')

    args = parser.parse_args()
    DATALOADERS,ESTS = config.get_configs(args.config)    
    #for data in DATALOADERS:
    #    first_element = data[1]
    #    break
    #print(first_element)
    
    if len(ESTS) == 0:
        raise Exception("No estimators provided")
    if len(DATALOADERS) == 0:
        raise ValueError('No dataset provided')

    for data in tqdm(DATALOADERS):
        print(ESTS)
        compare_estimators(ESTS,data)
        

    print('completed all experiments successfully!')



