import numpy as np
from numpy import concatenate as cat
from mcllm.data import *
from experiments.evaluation.model_util import ModelConfig
import os 
import mcllm.config
import torch
from mcllm.model.llm import *
from mcllm.model.competing_methods import *

checkpoint = "../../results/rowcol=1__reg=2__small/epoch=115-step=696.ckpt/checkpoint/mp_rank_00_model_states.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    d = torch.load(checkpoint)
else:
    d = torch.load(checkpoint,map_location=torch.device('cpu'))


module = mcllm.model.llm.TabLLM(**d['hyper_parameters'])
module.load_state_dict(d['module'])
module = module.to(device).eval()
print(d['hyper_parameters'])

ESTIMATORS = [
    ModelConfig('KNN',KNNImputer),
    ModelConfig('MeanImputer',MeanImputer),
    ModelConfig('MedianImputer',MedianImputer),
    ModelConfig('SoftImpute',SoftImpute),
    ModelConfig('McLLM',module)
]