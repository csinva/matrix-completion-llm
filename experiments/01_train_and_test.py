import torch
import mcllm.data
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
import os.path
import imodelsx.cache_save_utils

path_to_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def add_main_args(parser):
    """Caching uses the non-default values from argparse to name the saving directory.
    Changing the default arg an argument will break cache compatibility with previous runs.
    """

    # dataset args
    parser.add_argument(
        "--dataset_names_train", type=str, default=["breast_cancer"], help="names of training datasets"
    )
    parser.add_argument(
        "--dataset_names_test", type=str, default=["breast_cancer", "compas_two_year_clean"], help="names of test datasets"
    )
    parser.add_argument(
        "--subsample_frac", type=float, default=1, help="fraction of samples to use"
    )

    # evaluating misc args
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=join(path_to_repo, "results"),
        help="directory for saving",
    )
    parser.add_argument(
        "--frac_nans",
        type=float,
        default=0.1,
        help="fraction of values to set to nan",
    )

    # model args
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["mean_imputation", "median_imputation"],
        default="mean_imputation",
        help="name of model",
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

    # load model
    mc_model = mcllm.model.model.get_matrix_completion_model(args)

    # set up saving dictionary + save params file
    r = defaultdict(list)
    r.update(vars(args))
    r["save_dir_unique"] = save_dir_unique
    imodelsx.cache_save_utils.save_json(
        args=args, save_dir=save_dir_unique, fname="params.json", r=r
    )

    # fit
    if hasattr(mc_model, "fit"):
        mats_train, mats_test = mcllm.data.get_datasets_as_matrices(
            args.dataset_names_train)
        mc_model.fit(mats_train, mats_test)

    # test
    mats_train, mats_test = mcllm.data.get_datasets_as_matrices(
        args.dataset_names_test)
    errors = []
    for i in range(len(mats_test)):
        mat_test = mats_test[i]
        mat_test_nans = deepcopy(mat_test)
        idxs_nan = np.random.choice(
            np.arange(mat_test.size), size=int(args.frac_nans * mat_test.size), replace=False
        )
        mat_test_nans.ravel()[idxs_nan] = np.nan
        mat_imputed = mc_model.impute(mat_test_nans)
        mean_abs_err_normalized = np.mean(np.abs(mat_imputed.ravel()[idxs_nan] -
                                                 mat_test.ravel()[idxs_nan])) / np.mean(mat_test.ravel()[idxs_nan])
        errors.append(mean_abs_err_normalized)
    r['mean_abs_err_normalized_list'] = mean_abs_err_normalized
    r['avg_mean_abs_err_normalized_list'] = np.mean(mean_abs_err_normalized)
    logging.info(
        f'mean_abs_err_normalized {np.mean(mean_abs_err_normalized):0.4f}')

    # save results
    joblib.dump(
        r, join(save_dir_unique, "results.pkl")
    )  # caching requires that this is called results.pkl
    joblib.dump(mc_model, join(save_dir_unique, "model.pkl"))
    logging.info("Succesfully completed :)\n\n")
