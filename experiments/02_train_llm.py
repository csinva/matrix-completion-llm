import torch
import mcllm.data.imodels
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
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import imodels
import torch.nn.functional as F
import os.path
import imodelsx.cache_save_utils
import torch.utils.data as data

path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """
    parser.add_argument('--device', default='cuda',
                        type=str, help='Device to use')
    parser.add_argument('--num_epochs', default=100,
                        type=int, help='Number of epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=2048,
                        type=int, help='Batch size')
    parser.add_argument('--n_embed', default=10,
                        type=int, help='Embedding size')
    parser.add_argument('--dropout', default=0,
                        type=float, help='Dropout rate')
    parser.add_argument('--n_rows', default=50,
                        type=int, help='Number of rows')
    parser.add_argument('--n_columns', default=20,
                        type=int, help='Number of columns')
    parser.add_argument('--rank', default=5, type=int, help='Rank')
    parser.add_argument('--frac_nan_mask', default=0.1,
                        type=float, help='Fraction of NaN mask')
    parser.add_argument('--seed', default=13, type=int, help='Seed')
    parser.add_argument('--n_matrices', default=32768,
                        type=int, help='Number of matrices')
    parser.add_argument(
        "--save_dir",
        type=str,
        default=join(path_to_repo, "results"),
        help="directory for saving",
    )
    return parser


def add_computational_args(parser):
    """Arguments that only affect computation and not the results (shouldnt use when checking cache)"""
    parser.add_argument(
        "--use_cache",
        type=int,
        default=1,
        choices=[0, 1],
        help="whether to check for cache",
    )
    parser.add_argument(
        '--use_data_parallel',
        type=int,
        default=1,
        choices=[0, 1],
        help='whether to use data parallel',
    )
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

    llm = TabLLM(
        n_embed=args.n_embed, n_layers=3, n_heads=3, dropout=args.dropout).to(args.device)

    dataset = LowRankDataset(args.n_rows, args.n_columns, args.rank, args.frac_nan_mask,
                             seed=args.seed, length=args.n_matrices)

    llm = llm.to(args.device)
    if args.use_data_parallel:
        llm = torch.nn.DataParallel(llm, device_ids=[0, 1, 2, 3])
    optimizer = torch.optim.Adam(llm.parameters(), lr=args.lr)
    dataloader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)

    train_losses = []
    for i in range(args.num_epochs):
        train_loss = 0.0
        train_preds = []
        train_labels = []

        llm.train()

        for batch_num, (x_nan_t, x_clean_t, nan_mask_t) in enumerate(dataloader):
            x_nan_t = x_nan_t.to(args.device)
            x_clean_t = x_clean_t.to(args.device)
            nan_mask_t = nan_mask_t.to(args.device)
            att_mask_t = torch.ones_like(x_nan_t).to(args.device)

            pred = llm(x_nan_t, nan_mask_t, att_mask_t,
                       n_rows=args.n_rows, n_columns=args.n_columns)
            nan_loss = F.mse_loss(
                x_clean_t[nan_mask_t.bool()], pred[nan_mask_t.bool()])

            optimizer.zero_grad()
            nan_loss.backward()
            optimizer.step()

            train_loss += nan_loss.item() / nan_mask_t.sum().item()
            train_losses.append(train_loss)

        print(f'{i} -- Train loss {train_loss:0.6f}')

    # set up saving dictionary + save params file
    r = defaultdict(list)
    r.update(vars(args))
    r["save_dir_unique"] = save_dir_unique
    r['train_losses'] = train_losses

    # save results
    joblib.dump(
        r, join(save_dir_unique, "results.pkl")
    )  # caching requires that this is called results.pkl
    # joblib.dump(llm.state_dict, join(save_dir_unique, "model.pkl"))
    torch.save(llm.state_dict(), join(save_dir_unique, "model.pkl"))
    logging.info("Succesfully completed :)\n\n")
