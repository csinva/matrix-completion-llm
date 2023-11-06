import torch
from tqdm import tqdm
from mcllm.data.synthetic import LowRankDataset
from mcllm.model.llm import TabLLM
import mcllm.model.model
import argparse
from copy import deepcopy
import logging
import random
from collections import defaultdict
from os.path import join
import numpy as np
import joblib
import torch.nn.functional as F
import os.path
import imodelsx.cache_save_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data as data
import lightning as L
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.callbacks import ModelCheckpoint

path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # data args
    parser.add_argument('--n_rows_list', default=list(range(5, 21)),  # (5, 20)
                        type=int, nargs='+', help='Number of rows')
    parser.add_argument('--n_columns_list', default=list(range(5, 32)),  # (5, 20)
                        type=int, nargs='+', help='Number of columns')
    parser.add_argument('--rank_list', default=list(range(1, 6)),  # (1, 5)
                        type=int, nargs='+', help='Rank')
    parser.add_argument('--n_matrices_test', default=16384,
                        type=int, help='Number of matrices to put before printing (each matrix is newly generated)')
    parser.add_argument('--frac_nan_rand_mask_list', default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95],  # 0.025, 0.05, 0.1
                        type=float, nargs='+', help='Fraction of random NaN mask')
    parser.add_argument('--frac_nan_col_mask_list', default=[0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95],
                        type=float, nargs='+', help='Fraction of col-specific NaN mask (only)')
    parser.add_argument('--frac_col_vs_random', default=0.5,
                        type=float, help='Fraction of NaN masks that are col-specific vs random')

    # training args
    parser.add_argument('--lr', default=1e-3, type=float,  # 1e-3
                        help='Learning rate')
    parser.add_argument('--batch_size', default=1024,
                        type=int, help='Batch size')
    parser.add_argument('--seed', default=13, type=int, help='Seed')
    parser.add_argument('--num_epochs', default=100000,
                        type=int, help='Number of epochs')
    parser.add_argument("--save_dir", type=str,
                        default=join(path_to_repo, "results"),
                        help="directory for saving")

    # model args
    parser.add_argument('--n_layers', default=4,  # 8
                        type=int, help='Number of layers')
    parser.add_argument('--n_heads', default=4,  # 8
                        type=int, help='Number of heads')
    parser.add_argument('--n_embed', default=12,  # 96
                        type=int, help='Embedding size')
    parser.add_argument('--n_registers', default=1,
                        type=int, help='Number of registers (adds an extra row and column)')
    parser.add_argument('--dropout', default=0,
                        type=float, help='Dropout rate')
    parser.add_argument('--use_rowcol_attn', default=1, type=int, choices=[0, 1],
                        help='Whether to use row/column attention (otherwise use full attention)')

    return parser


def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldnt use when checking cache)"""

    parser.add_argument(
        '--check_val_every_n_epoch',
        type=int,
        default=4,
        help='how often to check validation loss',
    )
    parser.add_argument(
        "--use_cache",
        type=int,
        default=0,
        choices=[0, 1],
        help="whether to check for cache",
    )
    parser.add_argument('--device', default='cuda',
                        type=str, help='Device to use')
    return parser


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser()
    parser_without_computational_args = add_main_args(parser)
    parser = add_computational_args(
        deepcopy(parser_without_computational_args))
    args = parser.parse_args()

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    # set up saving directory + check for cache
    already_cached, save_dir_unique = imodelsx.cache_save_utils.get_save_dir_unique(
        parser, parser_without_computational_args, args, args.save_dir
    )

    if args.use_cache and already_cached:
        logging.info(f"cached version exists! Successfully skipping :)\n\n\n")
        exit(0)
    for k in sorted(vars(args)):
        logger.info("\t" + k + " " + str(vars(args)[k]))
    logging.info(f"\n\n\tsaving to " + save_dir_unique + "\n")

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # get data
    logging.info('generating data...')
    kwargs = {
        'm_list': args.n_rows_list,
        'n_list': args.n_columns_list,
        'rank_list': args.rank_list,
        'frac_nan_rand_mask_list': args.frac_nan_rand_mask_list,
        'frac_nan_col_mask_list': args.frac_nan_col_mask_list,
        'frac_col_vs_random': args.frac_col_vs_random,
        'use_rowcol_attn': args.use_rowcol_attn,
        'n_registers': args.n_registers,
    }
    dataset = LowRankDataset(
        seed=args.seed, length=args.batch_size * 16, randomize=True, **kwargs)
    dataset_test = LowRankDataset(
        seed=args.seed + 1, length=args.n_matrices_test, randomize=False, **kwargs)
    dataloader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=31)  # , num_workers=4)
    dataloader_test = data.DataLoader(
        dataset_test, batch_size=args.batch_size * 4, shuffle=False, num_workers=31)  # , num_workers=4)

    # set up saving dictionary + save params file
    r = defaultdict(list)
    r.update(vars(args))
    r["save_dir_unique"] = save_dir_unique
    imodelsx.cache_save_utils.save_json(
        r, save_dir=save_dir_unique, fname='params.json')

    # get model and optimizer
    logging.info('loading model.....')
    llm = TabLLM(
        n_embed=args.n_embed, n_layers=args.n_layers,
        n_heads=args.n_heads, dropout=args.dropout,
        use_pos_embeddings=not args.use_rowcol_attn,
        n_registers=args.n_registers,

        # training args
        learning_rate=args.lr,
        n_rows_list=args.n_rows_list,
        n_columns_list=args.n_columns_list,
    ).to(args.device)
    torch.set_float32_matmul_precision('medium')

    # train
    pl_logger = CSVLogger(save_dir_unique, name="logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir_unique, save_top_k=1, monitor="val_loss", mode='min')
    trainer = pl.Trainer(
        default_root_dir=save_dir_unique,
        max_epochs=args.num_epochs,
        logger=pl_logger, log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        #  callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)],
        strategy='deepspeed_stage_2',
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        enable_checkpointing=True,
        #  precision=16,
    )
    trainer.fit(model=llm, train_dataloaders=dataloader,
                val_dataloaders=dataloader_test)
