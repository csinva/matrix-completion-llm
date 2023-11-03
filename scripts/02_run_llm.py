from imodelsx import submit_utils
from os.path import dirname, join
import os.path
repo_dir = dirname(dirname(os.path.abspath(__file__)))

# decent test script
# CUDA_VISIBLE_DEVICES=0 python experiments/02_train_llm.py --n_rows_list 8 --n_columns_list 5 --rank_list 1 --lr 1e-3 --batch_size 4096 --frac_nan_mask 0.1 --n_layers 3 --n_heads 3 --n_embed 12 --n_registers 2 --use_rowcol_attn 1

# big expt with rowcol_attn and reg
# CUDA_VISIBLE_DEVICES=2,3 python experiments/02_train_llm.py --n_rows_list 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --n_columns_list 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --rank_list 1 2 3 4 5 --lr 1e-3 --batch_size 512 --n_layers 6 --n_heads 8 --n_embed 16 --n_registers 2 --frac_nan_mask 0.05 0.1 0.2 0.3 0.5 0.7 0.9 0.95 --use_rowcol_attn 1
# CUDA_VISIBLE_DEVICES=1,2,3 python experiments/02_train_llm.py --n_rows_list 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --n_columns_list 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --rank_list 1 2 3 4 5 --lr 1e-3 --batch_size 128 --n_layers 12 --n_heads 12 --n_embed 12 --n_registers 2 --frac_nan_mask 0.05 0.1 0.2 0.3 0.5 0.7 0.9 0.95 --use_rowcol_attn 1

# big expt with no rowcol_attn and no registers
# CUDA_VISIBLE_DEVICES=0,1 python experiments/02_train_llm.py --n_rows_list 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --n_columns_list 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --rank_list 1 2 3 4 5 --lr 1e-3 --batch_size 512 --n_layers 6 --n_heads 8 --n_embed 16 --n_registers 0 --frac_nan_mask 0.05 0.1 0.2 0.3 0.5 0.7 0.9 0.95 --use_rowcol_attn 0

# big expt with no rowcol_attn and 2 registers
# CUDA_VISIBLE_DEVICES=0,1 python experiments/02_train_llm.py --n_rows_list 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --n_columns_list 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --rank_list 1 2 3 4 5 --lr 1e-3 --batch_size 512 --n_layers 6 --n_heads 8 --n_embed 16 --n_registers 2 --frac_nan_mask 0.05 0.1 0.2 0.3 0.5 0.7 0.9 0.95 --use_rowcol_attn 0

# List of values to sweep over (sweeps over all combinations of these)
params_shared_dict = {
    'n_rows_list': [list(range(5, 21))],
    'n_columns_list': [list(range(5, 21))],
    'rank_list': [list(range(1, 6))],

    # computational
    'lr': [1e-3],

    # keep fixed
    'seed': [1],
    'use_cache': [0],
    'save_dir': [join(repo_dir, 'results')],
}

# List of tuples to sweep over (these values are coupled, and swept over together)
# Note: this is a dictionary so you shouldn't have repeated keys
params_coupled_dict = {}

# Args list is a list of dictionaries
# If you want to do something special to remove some of these runs, can remove them before calling run_args_list
args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)
submit_utils.run_args_list(
    args_list,
    script_name=join(repo_dir, 'experiments', '02_train_llm.py'),
    actually_run=True,
    gpu_ids=[0],
)
