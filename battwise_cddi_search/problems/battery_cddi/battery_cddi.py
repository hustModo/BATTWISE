"""
Battery MMD Problem Definition for EOH-Textgrad
Optimizes MMD calculation functions to maximize correlation with battery features
"""

import numpy as np
import re
import sys
import os
from typing import Tuple, Dict

# Add problems directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from gpt import evaluate_mmd_function


class BatteryMMDProblem:
    """
    Problem definition for battery MMD function optimization
    
    Objective: Maximize mean_sig (average absolute correlation between MMD and battery features)
    Search space: MMD calculation functions with different kernel strategies
    """
    
    def __init__(self, data_folder=None, model_filename=None, char_filename=None,
                 row_start=0, row_end=None, p_threshold=0.01, verbose=False):
        """
        Initialize Battery MMD Problem
        
        Args:
            data_folder: Path to battery data folder
            model_filename: Model/reference data file
            char_filename: Battery characteristics file
            row_start: Starting row for data selection
            row_end: Ending row for data selection
            p_threshold: P-value threshold for significance
            verbose: Print evaluation details
        """
        self.data_folder = data_folder
        self.model_filename = model_filename
        self.char_filename = char_filename
        self.row_start = row_start
        self.row_end = row_end
        self.p_threshold = p_threshold
        self.verbose = verbose
        
        self.name = "Battery_MMD_Optimization"
        self.objective = "maximize"  # Maximize mean_sig
        
    def evaluate_code(self, code: str) -> Tuple[float, Dict]:
        """
        Evaluate LLM-generated MMD function
        
        Args:
            code: Python code string containing calculate_mmd_vX function
            
        Returns:
            objective: mean_sig value (to be minimized, so we return negative)
            other_info: dict with detailed information
        """
        try:
            # Extract function from code
            namespace = {}
            
            # Add necessary imports to namespace
            exec("import numpy as np", namespace)
            exec("from scipy.spatial.distance import cdist", namespace)
            exec("import warnings; warnings.filterwarnings('ignore')", namespace)
            
            # Execute the code
            exec(code, namespace)
            
            # Find the calculate_mmd function
            mmd_func = None
            for name, obj in namespace.items():
                if callable(obj) and name.startswith('calculate_mmd_v'):
                    mmd_func = obj
                    break
            
            if mmd_func is None:
                print("Error: No calculate_mmd_vX function found in code")
                return float('inf'), {'error': 'No MMD function found', 'mean_sig': 0.0}
            
            # Evaluate the function
            mean_sig = evaluate_mmd_function(
                mmd_func,
                data_folder=self.data_folder,
                model_filename=self.model_filename,
                char_filename=self.char_filename,
                row_start=self.row_start,
                row_end=self.row_end,
                p_threshold=self.p_threshold,
                verbose=self.verbose
            )
            
            # Since we want to maximize mean_sig, return negative value
            # (EOH minimizes objective)
            objective = -mean_sig
            
            other_info = {
                'mean_sig': mean_sig,
                'status': 'success'
            }
            
            return objective, other_info
            
        except Exception as e:
            print(f"Error evaluating code: {e}")
            import traceback
            traceback.print_exc()
            
            return float('inf'), {
                'error': str(e),
                'mean_sig': 0.0,
                'status': 'failed'
            }
    
    def get_problem_description(self) -> str:
        """Return problem description for LLM prompts"""
        return """Battery Recycling: MMD-based Distribution Difference Quantification

**Problem Context:**
In battery recycling, we need to assess the performance differences of individual battery cells.
Internal electrochemical features are difficult to measure directly, but they manifest as
differences in charging curves (Current and Voltage data over time).

**Task:**
Design an optimal MMD (Maximum Mean Discrepancy) calculation function to quantify the
distribution differences between battery charging curves. The MMD values should strongly
correlate with battery electrochemical characteristics.

**Evaluation:**
The function is evaluated by computing the correlation between MMD loss values and battery
electrochemical features across multiple cells. The objective is to maximize the average
absolute Pearson correlation coefficient of statistically significant features (P < 0.01).

**Success Metric: mean_sig** (higher is better)
- mean_sig > 0.7: Extremely strong correlation
- mean_sig > 0.5: Strong correlation  
- mean_sig > 0.3: Moderate correlation
- mean_sig < 0.3: Weak correlation
"""

    def get_search_space_info(self) -> str:
        """Return information about the search space"""
        return """**Search Space:**

Key design choices for MMD calculation:
1. Bandwidth selection strategy (fixed vs adaptive)
2. Kernel multiplier factor (controls scale diversity)
3. Number of kernels (multi-scale analysis)
4. Distance metric and kernel function
5. Data preprocessing and normalization

**Constraints:**
- Must handle varying sample sizes
- Must be numerically stable
- Should avoid division by zero and overflow
- Computation should be efficient
"""


def get_problem_instance(config=None):
    """
    Factory function to create problem instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        BatteryMMDProblem instance
    """
    if config is None:
        config = {}
    
    # Default data paths (relative to this file)
    default_data_folder = os.path.join(
        os.path.dirname(__file__), 
        '..', 'battery_data', '200'
    )
    default_data_folder = os.path.abspath(default_data_folder)
    
    return BatteryMMDProblem(
        data_folder=config.get('data_folder', default_data_folder),
        model_filename=config.get('model_filename', 'Model_200.xls'),
        char_filename=config.get('char_filename', 'Electrochemical_indicators.xlsx'),
        row_start=config.get('row_start', 0),
        row_end=config.get('row_end', None),
        p_threshold=config.get('p_threshold', 0.01),
        verbose=config.get('verbose', False)
    )


# For testing
if __name__ == "__main__":
    print("Testing Battery MMD Problem...")
    
    problem = get_problem_instance({'verbose': True})
    
    print(f"\nProblem: {problem.name}")
    print(f"Objective: {problem.objective}")
    print("\n" + problem.get_problem_description())
    
    # Test with baseline code
    test_code = """
import numpy as np
from scipy.spatial.distance import cdist

def calculate_mmd_v1(xs, xt, fix_sigma=0, kernel_mul=2.0, kernel_num=5):
    s_num = xs.shape[0]
    t_num = xt.shape[0]
    n_sam = s_num + t_num
    
    x_new = np.vstack((xs, xt))
    L2_distance = cdist(x_new, x_new, metric='euclidean') ** 2
    
    if fix_sigma == 0:
        bandwidth = np.sum(L2_distance) / (n_sam**2 - n_sam)
    else:
        bandwidth = fix_sigma
    
    bandwidth /= (kernel_mul ** (kernel_num // 2))
    
    bandwidth_list = []
    for i in range(kernel_num):
        current_bw = bandwidth * (kernel_mul ** i)
        bandwidth_list.append(current_bw)
        
    kernel_val_sum = np.zeros((n_sam, n_sam))
    
    for bw in bandwidth_list:
        kernel_val = np.exp(-1 * L2_distance / bw)
        kernel_val_sum += kernel_val
        
    ss = np.mean(kernel_val_sum[:s_num, :s_num])
    tt = np.mean(kernel_val_sum[s_num:, s_num:])
    st = np.mean(kernel_val_sum[:s_num, s_num:])
    ts = np.mean(kernel_val_sum[s_num:, :s_num])
    
    MMD_loss = ss + tt - st - ts
    
    return MMD_loss
"""
    
    print("\nEvaluating baseline MMD function...")
    objective, info = problem.evaluate_code(test_code)
    
    print(f"\nResults:")
    print(f"  Objective (negative mean_sig): {objective:.4f}")
    print(f"  Mean_sig (actual): {info['mean_sig']:.4f}")
    print(f"  Status: {info['status']}")
    
    if info['mean_sig'] > 0:
        print("\n✅ Problem definition working correctly!")
    else:
        print("\n⚠️ Warning: mean_sig = 0. Check data paths.")
