import itertools
import os
import warnings
from functools import partial
from os.path import dirname
from os.path import join as oj
from typing import Any, Dict, Tuple
import numpy as np



class ModelConfig:
    '''
    Specify matrix completion model confguration
    name: str
        Name of model
    cls: 
        unitialized model class
    vary_param: str 
        value of parameter to be varied
    vary_param_val: Any
        value of parameter to be varied

    '''
    
    def __init__(self,
                 name: str, cls,
                 vary_param: str = None, vary_param_val: Any = None):
        self.name = name
        self.cls = cls
        self.vary_param = vary_param
        self.vary_param_val = vary_param_val
    
    def __repr__(self):
        return self.name