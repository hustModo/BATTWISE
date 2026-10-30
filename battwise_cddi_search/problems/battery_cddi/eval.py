"""
Battery MMD Evaluation Script
Called by EOH framework to evaluate LLM-generated MMD functions
"""

import sys
import os
import numpy as np
import pandas as pd
import glob
import re
import importlib.util
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Get arguments from command line
root_dir = sys.argv[1]
mood = sys.argv[2]  # 'train' or 'val'
case_num_str = sys.argv[3]  # Case selector: CV, first_100s, first_200s, first_1000s
island_id = int(sys.argv[4])  # Response ID
cell_filter_arg = sys.argv[5] if len(sys.argv) > 5 else ""


def parse_cell_filter(raw):
    if not raw:
        return None
    nums = [int(x) for x in re.findall(r'\d+', str(raw))]
    return set(nums) if nums else None


CELL_FILTER = parse_cell_filter(cell_filter_arg)

# Add paths
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'problems', 'battery_cddi'))

# Case configuration mapping
# 索引映射: 0=CV_100, 1=first_100s, 2=first_200s, 3=first_500s, 4=AII_200, 5=CC_200
CASE_CONFIGS = {
    'CV_100': {
        'folder': 'CV_100',
        'model_file': '100_Model_CV_process.xlsx',
        'char_file': 'Charging_curve_characteristics.xlsx',
        'row_start': 0,
        'row_end': None,
        'p_threshold': 1
    },
    'first_100s': {
        'folder': 'first_100s',
        'model_file': 'Model_first_100s.xls',
        'char_file': 'Charging_curve_characteristics.xlsx',
        'row_start': 0,
        'row_end': None,
        'p_threshold': 1
    },
    'first_200s': {
        'folder': 'first_200s',
        'model_file': 'Model_first_200s.xls',
        'char_file': 'Charging_curve_characteristics.xlsx',
        'row_start': 0,
        'row_end': None,
        'p_threshold': 1
    },
    'first_500s': {
        'folder': 'first_500s',
        'model_file': 'Model_first_500s.xls',
        'char_file': 'Charging_curve_characteristics.xlsx',
        'row_start': 0,
        'row_end': None,
        'p_threshold': 1
    },
    'AII_200': {
        # Current repository stores the All-200 refinement curves under
        # AII_200_new (71 cells, matching first_200s). Historical logs used
        # AII_200, but that directory is not present in this project copy.
        'folder': 'AII_200_new',
        'model_file': 'Model_200.xls',
        'char_file': 'Charging_curve_characteristics.xlsx',
        'row_start': 0,
        'row_end': None,
        'p_threshold': 1
    },
    'CC_200': {
        'folder': 'CC_200',
        'model_file': '200_Model_CC_process.xlsx',
        'char_file': 'Charging_curve_characteristics.xlsx',
        'row_start': 0,
        'row_end': None,
        'p_threshold': 1
    }
}

# Select case configuration based on case_num_str
try:
    # Parse case_num_str (could be "0", "CV", "first_100s", etc.)
    case_num_str = case_num_str.strip()
    
    # If numeric, map to case names
    if case_num_str.isdigit():
        case_index = int(case_num_str)
        # 索引映射: 0=CV_100, 1=first_100s, 2=first_200s, 3=first_500s, 4=AII_200, 5=CC_200
        case_names = ['CV_100', 'first_100s', 'first_200s', 'first_500s', 'AII_200', 'CC_200']
        if 0 <= case_index < len(case_names):
            selected_case = case_names[case_index]
        else:
            print(f"[!] Warning: Invalid case index {case_index}, defaulting to CV_100")
            selected_case = 'CV_100'
    else:
        # Direct case name
        selected_case = case_num_str
    
    # Get case configuration
    if selected_case in CASE_CONFIGS:
        case_config = CASE_CONFIGS[selected_case]
        DATA_FOLDER = os.path.join(root_dir, 'problems', 'battery_data', case_config['folder'])
        MODEL_FILENAME = case_config['model_file']
        CHAR_FILENAME = case_config['char_file']
        ROW_START = case_config['row_start']
        ROW_END = case_config['row_end']
        P_VALUE_THRESHOLD = case_config['p_threshold']
        print(f"[*] Selected Case: {selected_case}")
    else:
        print(f"[!] Warning: Unknown case '{selected_case}', defaulting to CV_100")
        case_config = CASE_CONFIGS['CV_100']
        DATA_FOLDER = os.path.join(root_dir, 'problems', 'battery_data', case_config['folder'])
        MODEL_FILENAME = case_config['model_file']
        CHAR_FILENAME = case_config['char_file']
        ROW_START = case_config['row_start']
        ROW_END = case_config['row_end']
        P_VALUE_THRESHOLD = case_config['p_threshold']
        
except Exception as e:
    # Fallback to CV_100 on any error
    print(f"[!] Error parsing case configuration: {e}, defaulting to CV_100")
    case_config = CASE_CONFIGS['CV_100']
    DATA_FOLDER = os.path.join(root_dir, 'problems', 'battery_data', case_config['folder'])
    MODEL_FILENAME = case_config['model_file']
    CHAR_FILENAME = case_config['char_file']
    ROW_START = case_config['row_start']
    ROW_END = case_config['row_end']
    P_VALUE_THRESHOLD = case_config['p_threshold']


def load_data(file_path):
    """Helper function: Load CSV or Excel file"""
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file_path)
        else:
            return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def extract_cell_number(file_path):
    """Extract cell number from filename"""
    filename = os.path.basename(file_path)
    match = re.search(r'Cell(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return float('inf')


if __name__ == "__main__":
    print("[*] Battery MMD Evaluation Running...")
    print(f"[*] Case: {case_num_str}")
    print(f"[*] Data Folder: {DATA_FOLDER}")
    print(f"[*] Model File: {MODEL_FILENAME}")
    print(f"[*] Island ID: {island_id}")
    if CELL_FILTER:
        print(f"[*] Cell filter: {sorted(CELL_FILTER)}")
    
    try:
        # Step 1: Load LLM-generated MMD function
        gpt_file_path = os.path.join(root_dir, 'baselines', 'eoh', 'gpt_code', f'gpt_code_{island_id}.py')
        
        if not os.path.exists(gpt_file_path):
            print(f"[!] Error: Generated code file not found: {gpt_file_path}")
            print(f"Error code: {float('inf')}")
            sys.exit(1)
        
        # Dynamically import the generated MMD function
        spec = importlib.util.spec_from_file_location(f"gpt_code_{island_id}", gpt_file_path)
        gpt_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gpt_module)
        
        # Find the calculate_mmd function
        mmd_func = None
        for name in dir(gpt_module):
            if name.startswith('calculate_mmd_v'):
                mmd_func = getattr(gpt_module, name)
                print(f"[*] Loaded function: {name}")
                break
        
        if mmd_func is None:
            print("[!] Error: No calculate_mmd_vX function found")
            print(f"Error code: {float('inf')}")
            sys.exit(1)
        
        # Step 2: Load Model (reference) data
        model_path = os.path.join(DATA_FOLDER, MODEL_FILENAME)
        df_model = load_data(model_path)
        
        if df_model is None:
            print(f"[!] Error: Cannot load Model file {model_path}")
            print(f"Error code: {float('inf')}")
            sys.exit(1)
        
        # Extract Current and Voltage
        if ROW_END is not None:
            xt = df_model[['Current', 'Voltage']].iloc[ROW_START:ROW_END].values
        else:
            xt = df_model[['Current', 'Voltage']].iloc[ROW_START:].values
        
        print(f"[*] Model samples: {xt.shape[0]}")
        
        # Step 3: Compute MMD for all Cell files
        search_pattern = os.path.join(DATA_FOLDER, '*Cell*.*')
        cell_files = glob.glob(search_pattern)
        cell_files.sort(key=extract_cell_number)
        if CELL_FILTER:
            cell_files = [p for p in cell_files if extract_cell_number(p) in CELL_FILTER]
        
        print(f"[*] Found {len(cell_files)} Cell files")
        
        mmd_results = []
        
        for cell_file in cell_files:
            df_cell = load_data(cell_file)
            if df_cell is not None:
                try:
                    # Extract Current and Voltage
                    if ROW_END is not None:
                        xs = df_cell[['Current', 'Voltage']].iloc[ROW_START:ROW_END].values
                    else:
                        xs = df_cell[['Current', 'Voltage']].iloc[ROW_START:].values
                    
                    # Call LLM-generated MMD function
                    loss = mmd_func(xs, xt)
                    
                    cell_num = extract_cell_number(cell_file)
                    mmd_results.append({'Cell_num': cell_num, 'MMD_Loss': loss})
                    
                except Exception as e:
                    print(f"[!] Error processing {os.path.basename(cell_file)}: {e}")
                    continue
        
        if len(mmd_results) == 0:
            print("[!] Error: No valid MMD results computed")
            print(f"Error code: {float('inf')}")
            sys.exit(1)
        
        print(f"[*] MMD computation complete: {len(mmd_results)} samples")
        
        # Step 4: Load characteristics and compute correlation
        char_path = os.path.join(DATA_FOLDER, CHAR_FILENAME)
        df_char = load_data(char_path)
        
        if df_char is None:
            print(f"[!] Error: Cannot load characteristics file")
            print(f"Error code: {float('inf')}")
            sys.exit(1)
        
        # Handle multi-level header
        if 'Cell_num' not in df_char.columns:
            if 'Cell_num' in df_char.iloc[0].astype(str).values:
                df_char.columns = df_char.iloc[0]
                df_char = df_char.iloc[1:].reset_index(drop=True)
        
        df_char['Cell_num'] = pd.to_numeric(df_char['Cell_num'], errors='coerce')
        
        # Merge MMD results with characteristics
        df_mmd = pd.DataFrame(mmd_results)
        df_merged = pd.merge(df_mmd, df_char, on='Cell_num', how='inner')
        
        print(f"[*] Matched {len(df_merged)} samples for correlation analysis")
        
        # Compute correlation
        cols_to_analyze = [col for col in df_char.columns if col != 'Cell_num']
        significant_corrs = []
        
        for col in cols_to_analyze:
            y = pd.to_numeric(df_merged[col], errors='coerce')
            
            if y.notna().sum() > 2:
                x = df_merged['MMD_Loss']
                valid_mask = ~np.isnan(x) & ~np.isnan(y)
                
                if np.sum(valid_mask) > 2:
                    r, p = pearsonr(x[valid_mask], y[valid_mask])
                    
                    if p < P_VALUE_THRESHOLD:
                        significant_corrs.append(abs(r))
        
        # Step 5: Compute mean_sig
        if len(significant_corrs) > 0:
            mean_sig = np.mean(significant_corrs)
        else:
            mean_sig = 0.0
        
        # Since EOH minimizes, return negative mean_sig
        objective = -mean_sig
        
        print(f"[*] Significant features: {len(significant_corrs)}")
        print(f"[*] mean_sig: {mean_sig:.4f}")
        print(f"[*] Objective (negative): {objective:.4f}")
        
        # Print in EOH expected format
        print(f"Error code: {objective}")
        
    except Exception as e:
        print(f"[!] Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        print(f"Error code: {float('inf')}")
