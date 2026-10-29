import numpy as np
from scipy.spatial.distance import cdist

def calculate_mmd_v1(xs: np.ndarray, xt: np.ndarray, fix_sigma: float = 0, kernel_mul: float = 2.0, kernel_num: int = 5) -> float:
    """
    Baseline MMD calculation with standard multi-kernel approach
    """
    s_num = xs.shape[0]
    t_num = xt.shape[0]
    n_sam = s_num + t_num
    
    # Combine data
    x_new = np.vstack((xs, xt))
    
    # Compute squared Euclidean distance matrix
    L2_distance = cdist(x_new, x_new, metric='euclidean') ** 2
    
    # Compute bandwidth
    if fix_sigma == 0:
        bandwidth = np.sum(L2_distance) / (n_sam**2 - n_sam)
    else:
        bandwidth = fix_sigma
    
    bandwidth /= (kernel_mul ** (kernel_num // 2))
    
    # Generate bandwidth list
    bandwidth_list = []
    for i in range(kernel_num):
        current_bw = bandwidth * (kernel_mul ** i)
        bandwidth_list.append(current_bw)
        
    kernel_val_sum = np.zeros((n_sam, n_sam))
    
    # Compute multi-kernel Gaussian matrix sum
    for bw in bandwidth_list:
        kernel_val = np.exp(-1 * L2_distance / bw)
        kernel_val_sum += kernel_val
        
    # Compute MMD Loss
    ss = np.mean(kernel_val_sum[:s_num, :s_num])
    tt = np.mean(kernel_val_sum[s_num:, s_num:])
    st = np.mean(kernel_val_sum[:s_num, s_num:])
    ts = np.mean(kernel_val_sum[s_num:, :s_num])
    
    MMD_loss = ss + tt - st - ts
    
    return MMD_loss
