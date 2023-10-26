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
# python experiments/02_train_llm.py --use_data_parallel 0 --rank 1 --n_matrices=1024
path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # data args
    parser.add_argument('--n_rows_list', default=range(5, 20),
                        type=list, help='Number of rows')
    parser.add_argument('--n_columns_list', default=range(5, 20),
                        type=list, help='Number of columns')
    parser.add_argument('--rank_list', default=range(1, 5),
                        type=int, help='Rank')
    parser.add_argument('--n_matrices_test', default=32768,
                        type=int, help='Number of matrices to put before printing (each matrix is newly generated)')

    # training args
    parser.add_argument('--frac_nan_mask', default=[0.025, 0.05, 0.1],
                        type=float, help='Fraction of NaN mask')
    parser.add_argument('--seed', default=13, type=int, help='Seed')
    parser.add_argument('--num_epochs', default=100000,
                        type=int, help='Number of epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=256,
                        type=int, help='Batch size')

    # model args
    parser.add_argument('--n_layers', default=8,
                        type=int, help='Number of layers')
    parser.add_argument('--n_heads', default=8,
                        type=int, help='Number of heads')
    parser.add_argument('--n_embed', default=96,
                        type=int, help='Embedding size')
    parser.add_argument('--dropout', default=0,
                        type=float, help='Dropout rate')

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
        '--save_freq',
        type=int,
        default=10,
        help='how often to save model and results (only saves if improvement)',
    )
    parser.add_argument(
        "--use_cache",
        type=int,
        default=0,
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
    dataset = LowRankDataset(args.n_rows_list, args.n_columns_list, args.rank_list, args.frac_nan_mask,
                             seed=args.seed, length=args.batch_size * 1, randomize=True)
    dataset_test = LowRankDataset(args.n_rows_list, args.n_columns_list, args.rank_list, args.frac_nan_mask,
                                  seed=args.seed + 1, length=args.n_matrices_test, randomize=False)
    dataloader = data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)
    dataloader_test = data.DataLoader(
        dataset_test, batch_size=args.batch_size * 2, shuffle=False)

    # set up saving dictionary + save params file
    r = defaultdict(list)
    r.update(vars(args))
    r["save_dir_unique"] = save_dir_unique

    # get model and optimizer
    logging.info('loading model.....')
    llm = TabLLM(
        n_embed=args.n_embed, n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout).to(args.device)
    if args.use_data_parallel:
        llm = torch.nn.DataParallel(llm, device_ids=[0, 1, 2, 3])
    optimizer = torch.optim.Adam(llm.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    train_losses = []
    test_losses = []
    best_saved_test_loss = 1e10
    logging.info('starting training...')
    for i in tqdm(range(args.num_epochs)):

        # train and compute train loss
        llm.train()
        train_loss = 0.0
        n_masked = 0
        for batch_num, (x_nan_t, x_clean_t, nan_mask_t, att_mask_t) in enumerate(dataloader):
            x_nan_t = x_nan_t.to(args.device)
            x_clean_t = x_clean_t.to(args.device)
            nan_mask_t = nan_mask_t.to(args.device)
            att_mask_t = att_mask_t.to(args.device)
            # with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            pred = llm(x_nan_t, nan_mask_t, att_mask_t,
                       n_rows=max(args.n_rows_list), n_columns=max(args.n_columns_list))
            nan_loss = F.mse_loss(
                x_clean_t[nan_mask_t.bool()], pred[nan_mask_t.bool()], reduction='mean')

            optimizer.zero_grad()
            nan_loss.backward()
            optimizer.step()

            train_loss += (
                torch.abs(x_clean_t[nan_mask_t.bool()] -
                          pred[nan_mask_t.bool()])
            ).sum().item()  #
            n_masked += nan_mask_t.sum().item()

        train_loss /= n_masked
        # print(f'{i} -- Train loss {train_loss}')
        train_losses.append(train_loss)

        # compute test loss
        llm.eval()
        test_loss = 0.0
        n_masked = 0
        for batch_num, (x_nan_t, x_clean_t, nan_mask_t, att_mask_t) in enumerate(dataloader_test):
            x_nan_t = x_nan_t.to(args.device)
            x_clean_t = x_clean_t.to(args.device)
            nan_mask_t = nan_mask_t.to(args.device)
            att_mask_t = att_mask_t.to(args.device)

            pred = llm(x_nan_t, nan_mask_t, att_mask_t,
                       n_rows=max(args.n_rows_list), n_columns=max(args.n_columns_list))
            test_loss += (
                torch.abs(x_clean_t[nan_mask_t.bool()] -
                          pred[nan_mask_t.bool()])
            ).sum().item()  #
            n_masked += nan_mask_t.sum().item()

        test_loss /= n_masked
        scheduler.step(test_loss)
        if i == 0:
            print('~Baseline loss', torch.abs(
                x_clean_t[nan_mask_t.bool()]).mean().item())
        print(f'{i} -- Test loss {test_loss}')
        test_losses.append(test_loss)

        if i > 5 and i % args.save_freq == 0:
            os.makedirs(save_dir_unique, exist_ok=True)
            r['train_losses'] = train_losses
            r['test_losses'] = test_losses

            # save results
            if test_loss < best_saved_test_loss:
                best_saved_test_loss = test_loss
                r['best_saved_test_loss'] = best_saved_test_loss
                r['best_saved_idx'] = i
                joblib.dump(
                    r, join(save_dir_unique, "results.pkl")
                )
                torch.save(llm.state_dict(), join(
                    save_dir_unique, "model.pkl"))

    logging.info("Succesfully completed :)\n\n")
