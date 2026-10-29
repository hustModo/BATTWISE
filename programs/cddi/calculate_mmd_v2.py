import numpy as np
from scipy.linalg import eigh

def calculate_mmd_v2(xs: np.ndarray, xt: np.ndarray, fix_sigma: float = 0, kernel_mul: float = 2.0, kernel_num: int = 5) -> float:
    """
    Spectral-regularized MMD with energy-based bandwidth adaptation and phase-sensitive feature alignment.
    Uses Nyström approximation for stable eigen-decomposition and voltage-current coherence weighting.
    """
    # Input validation and preprocessing
    if xs.size == 0 or xt.size == 0:
        return 0.0
    
    # Phase-sensitive feature alignment (voltage-current coherence)
    phase_weights = np.array([0.4, 0.6])  # Current/Voltage importance
    xs = xs * phase_weights
    xt = xt * phase_weights
    
    # Energy-based bandwidth selection
    combined = np.vstack([xs, xt])
    if fix_sigma == 0:
        # Spectral energy thresholding
        cov = np.cov(combined.T)
        eigvals = eigh(cov, eigvals_only=True, subset_by_index=[0,0])
        sigma = np.sqrt(np.max(eigvals)) * 0.25  # Energy-based scaling
    else:
        sigma = fix_sigma
    
    # Regularized kernel spectrum
    bandwidths = []
    for i in range(kernel_num):
        b = sigma * (kernel_mul ** i)
        if b > 1e-8:
            bandwidths.append(b)
    
    # Nyström-approximated MMD calculation
    n_xs = len(xs)
    n_xt = len(xt)
    mmd_total = 0.0
    
    for b in bandwidths:
        # Random subset for Nyström approximation
        subset_size = min(50, n_xs+n_xt)
        idx = np.random.choice(n_xs+n_xt, subset_size, replace=False)
        subset = combined[idx]
        
        # Approximate kernel matrices
        K = np.exp(-np.sum((combined[:, None] - subset[None])**2, axis=-1)/(2*b**2))
        K_subset = K[idx]
        
        # Eigen-decomposition with regularization
        try:
            eigvals, eigvecs = eigh(K_subset)
            eigvals = np.maximum(eigvals, 1e-8)
            proj = K @ (eigvecs / np.sqrt(eigvals))
            
            # Spectral MMD components
            xs_proj = proj[:n_xs]
            xt_proj = proj[n_xs:]
            mmd = (np.mean(xs_proj**2, axis=0) + np.mean(xt_proj**2, axis=0) 
                  - 2*np.mean(xs_proj * xt_proj, axis=0)).sum()
            mmd_total += np.sqrt(max(mmd, 0))
        except:
            continue
    
    # Energy-normalized output
    return float(mmd_total / len(bandwidths)) if bandwidths else 0.0
