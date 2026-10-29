# ============================================================================
# Baseline MMD_cal_new — multi-bandwidth Gaussian MMD.
#
# This is the "factory default" that the Designer/Compute pipeline injects
# replacements into. Restored after the timeout regression test pollutes
# this file with a hanging stub.
#
# Reference implementation: dianchi03/mmd_tools.py (the original
# `MMD_cal` function) — renamed to `MMD_cal_new` so it matches the
# function name the rest of the pipeline expects.
# ============================================================================

import numpy as np
from scipy.spatial.distance import cdist


def MMD_cal_new(xs, xt, fix_sigma, kernel_mul, kernel_num):
    """Baseline multi-bandwidth Gaussian MMD between source `xs` and target `xt`.

    Args:
        xs: source data, shape (n_s, d).
        xt: target data, shape (n_t, d).
        fix_sigma: if 0, bandwidth is the median heuristic; else use this value.
        kernel_mul: bandwidth multiplicative factor.
        kernel_num: number of Gaussian kernels in the multi-kernel sum.

    Returns:
        Scalar MMD loss (float).
    """
    s_num = xs.shape[0]
    t_num = xt.shape[0]
    n_sam = s_num + t_num

    x_new = np.vstack((xs, xt))
    L2_distance = cdist(x_new, x_new, metric="euclidean") ** 2

    if fix_sigma == 0:
        bandwidth = np.sum(L2_distance) / (n_sam ** 2 - n_sam)
    else:
        bandwidth = fix_sigma
    bandwidth /= kernel_mul ** (kernel_num // 2)

    bandwidth_list = [bandwidth * (kernel_num ** i) for i in range(kernel_num)]

    kernel_val_sum = np.zeros((n_sam, n_sam))
    for bw in bandwidth_list:
        kernel_val_sum += np.exp(-L2_distance / bw)

    ss = np.mean(kernel_val_sum[:s_num, :s_num])
    tt = np.mean(kernel_val_sum[s_num:, s_num:])
    st = np.mean(kernel_val_sum[:s_num, s_num:])
    ts = np.mean(kernel_val_sum[s_num:, :s_num])

    return ss + tt - st - ts
