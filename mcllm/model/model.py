import numpy as np


def get_matrix_completion_model(args):
    if args.model_name == 'mean_imputation':
        model = MeanImputer()
    elif args.model_name == 'median_imputation':
        model = MedianImputer()
    else:
        raise ValueError(f'Invalid model_name: {args.model_name}')
    return model


class MeanImputer:
    def impute(self, m: np.ndarray):
        '''Impute each nan value with the mean of its column
        '''
        col_mean = np.nanmean(m, axis=0)
        inds = np.where(np.isnan(m))
        m[inds] = np.take(col_mean, inds[1])
        return m


class MedianImputer:
    def impute(self, m: np.ndarray):
        '''Impute each nan value with the median of its column
        '''
        col_median = np.nanmedian(m, axis=0)
        inds = np.where(np.isnan(m))
        m[inds] = np.take(col_median, inds[1])
        return m
