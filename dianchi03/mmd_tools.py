import numpy as np
from scipy.spatial.distance import cdist

def MMD_cal(xs, xt, fix_sigma, kernel_mul, kernel_num):
    """
    计算最大均值差异 (MMD)
    :param xs: 源域数据 (n_samples, n_features)
    :param xt: 目标域数据 (n_samples, n_features)
    :param fix_sigma: 固定 sigma 值
    :param kernel_mul: 带宽乘数因子
    :param kernel_num: 核的数量
    :return: MMD loss
    """
    s_num = xs.shape[0]
    t_num = xt.shape[0]
    n_sam = s_num + t_num
    
    # 合并数据
    x_new = np.vstack((xs, xt))
    
    # 计算欧氏距离平方矩阵
    L2_distance = cdist(x_new, x_new, metric='euclidean') ** 2
    
    # 计算带宽 (bandwidth)
    if fix_sigma == 0:
        bandwidth = np.sum(L2_distance) / (n_sam**2 - n_sam)
    else:
        bandwidth = fix_sigma
    
    bandwidth /= (kernel_mul ** (kernel_num // 2))
    
    bandwidth_list = []
    for i in range(kernel_num):
        current_bw = bandwidth * (kernel_num ** i)
        bandwidth_list.append(current_bw)
        
    kernel_val_sum = np.zeros((n_sam, n_sam))
    
    # 计算多核高斯矩阵和
    for bw in bandwidth_list:
        kernel_val = np.exp(-1 * L2_distance / bw)
        kernel_val_sum += kernel_val
        
    # 计算 MMD Loss
    ss = np.mean(kernel_val_sum[:s_num, :s_num])
    tt = np.mean(kernel_val_sum[s_num:, s_num:])
    st = np.mean(kernel_val_sum[:s_num, s_num:])
    ts = np.mean(kernel_val_sum[s_num:, :s_num])
    
    MMD_loss = ss + tt - st - ts
    
    return MMD_loss