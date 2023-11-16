from typing import List
import torch.utils.data as data
import numpy as np
import torch


def get_register_mask(m_max, n_max, n_registers):
    '''Returns flattened mask that is 0 everywhere except for the registers.
    Note: registers are stored at the right / bottom of the sequence
    '''
    register_mask = torch.zeros((m_max + n_registers,
                                 n_max + n_registers))
    if n_registers > 0:
        register_mask[-n_registers:] = 1
        register_mask[:, -n_registers:] = 1
    return register_mask.flatten()


def get_att_mask(m_max, n_max, n_registers, m, n, use_rowcol_attn: bool):
    '''Create attention mask of size (seq_len, seq_len)
    Attention is set to 0 except for registers and points in the same row/col (up to m, n)
    Note: registers are stored at the right / bottom of the sequence
        Each row/col can attend to its own registers, and registers can attend to each other
        in addition to their own row/col
    '''
    seq_len = (m_max + n_registers) * \
        (n_max + n_registers)
    m_max_with_reg = m_max + n_registers

    # basic attn_mask
    att_mask_kernel = torch.zeros((seq_len, seq_len))
    # seq_len_before_reg = (m_max + n_registers) * n_max
    # everything attends to registers (also registers attend to each other)
    # att_mask_kernel[seq_len_before_reg:] = 1
    # att_mask_kernel[:, seq_len_before_reg:] = 1

    if use_rowcol_attn:
        for i in range(seq_len):
            r_idx_row = i // m_max_with_reg
            c_idx_row = i % m_max_with_reg
            for j in range(seq_len):
                r_idx_col = j // m_max_with_reg
                c_idx_col = j % m_max_with_reg

                # everything attends to points in the same row/col
                if r_idx_row == r_idx_col or c_idx_row == c_idx_col:
                    att_mask_kernel[i, j] = 1

                # register attention
                # if c_idx_col >= n_max:
                    # att_mask_kernel[i, j] = 1
    else:
        # attention mask for full attention
        att_mask_kernel[:m*n, :m*n] = 1
    return att_mask_kernel


def get_nan_mask(
    m_max, n_max, n_registers, m, n,
    frac_nan_rand_mask_list, frac_nan_col_mask_list, frac_col_vs_random,
    rng
):
    '''
    nan mask - randomly mask some frac (1 means nan)
    only mask values in first m rows and first n cols
    '''
    nan_mask = np.zeros((m_max + n_registers,
                        n_max + n_registers))
    use_col_mask = rng.binomial(1, frac_col_vs_random)
    if use_col_mask:
        # set fraction of a single random column to nan
        frac_nan_col_mask = rng.choice(frac_nan_col_mask_list)
        col_idx = rng.choice(n)
        nan_mask[:m, col_idx] = rng.binomial(
            1, frac_nan_col_mask, size=(m, 1)).flatten()
    else:
        frac_nan_rand_mask = rng.choice(frac_nan_rand_mask_list)
        nan_mask[:m, :n] = rng.binomial(1, frac_nan_rand_mask, size=(m, n))
    return torch.Tensor(nan_mask).flatten()
