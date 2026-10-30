import hydra
import logging 
import os
from pathlib import Path
import subprocess
from utils.utils import init_client
import random
from multiprocessing import Pool, Manager
import time
from omegaconf import OmegaConf
import datetime

ROOT_DIR = os.getcwd() # 获取当前目录
import sys
python_executable = sys.executable
logging.basicConfig(level=logging.INFO) # 配置日志记录系统
@hydra.main(version_base=None, config_path="cfg", config_name="config_battery_cddi")

def main(cfg):
    workspace_dir = Path.cwd()
    
    # 在outputs/output/下创建时间戳子目录
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(ROOT_DIR, 'outputs', 'output', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # 添加文件日志处理器，将日志也保存到时间戳目录
    log_file = os.path.join(output_dir, 'main.log')
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Set logging level
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")
    logging.info(f"Output Directory: {output_dir}")
    logging.info(f"Log File: {log_file}")
    logging.info(f"Using LLM: {cfg.model}")
    logging.info(f"Using Algorithm: {cfg.algorithm}")

    # 初始化LLM客户端
    init_client(cfg)
    
    # Canonical BATTWISE reflective-search entry point.  The legacy value is
    # accepted so historical Hydra commands remain reproducible.
    if cfg.algorithm in {"reflective_metric_search", "eoh_textgrad"}:
        from reflective_metric_search import ReflectiveMetricSearch as LHH
    elif cfg.algorithm == "eoh":
        from baselines.eoh.eoh_adapter import EoH as LHH
    else:
        raise ValueError(
            f"Unsupported algorithm: {cfg.algorithm}. "
            "Use 'reflective_metric_search' or the attributed 'eoh' baseline."
        )

    # Main algorithm
    logging.info(f"Case num: {cfg.case_num}")
    if getattr(cfg, "eoh_cell_filter", None):
        logging.info(f"EOH cell filter: {cfg.eoh_cell_filter}")
    
    # 初始化算法实例
    # 确保case_num是列表格式（OmegaConf的ListConfig需要转换为普通list）
    from omegaconf import ListConfig
    if isinstance(cfg.case_num, (list, ListConfig)):
        # 转换为普通Python列表
        case_num = list(cfg.case_num)
    else:
        case_num = [cfg.case_num]
    
    # 使用固定的outputs/output目录
    lhh = LHH(cfg, ROOT_DIR, case_num, output_dir=output_dir)
    
    try:
        best_code_overall, best_code_path_overall, best_obj_overall = lhh.evolve()
        logging.info(f"Best Code Overall: {best_code_overall[:100] if best_code_overall else 'None'}...")
        logging.info(f"Best Code Path Overall: {best_code_path_overall}")
        logging.info(f"Best Objective Overall: {best_obj_overall}")
    except Exception as e:
        logging.error(f"Error during evolution: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

    # Save the best code in the desired format
    try:
        best_code_filename = f"best_code_{cfg.problem.problem_name}.txt"
        file_path = os.path.join(output_dir, best_code_filename)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write("```python\n")
            file.write(best_code_overall if best_code_overall else "# No code generated")
            file.write("\n```\n")
        logging.info(f"Saved best code to: {file_path}")
    except Exception as e:
        logging.error(f"Error saving best code: {e}")
    
    # Save to problem directory for evaluation
    try:
        problem_gpt_path = os.path.join(ROOT_DIR, 'problems', cfg.problem.problem_name, 'gpt.py')
        with open(problem_gpt_path, 'w', encoding='utf-8') as file:
            file.write(best_code_overall if best_code_overall else "# No code generated")
            file.write('\n')
        logging.info(f"Saved code to problem directory: {problem_gpt_path}")
    except Exception as e:
        logging.error(f"Error saving to problem directory: {e}")
    
    # Run validation (currently commented out)
    # test_script = os.path.join(ROOT_DIR, 'problems', cfg.problem.problem_name, 'eval.py')
    # test_script_stdout = os.path.join(output_dir, "validation_output.txt")
    # logging.info(f"Running validation script: {test_script}")
    # try:
    #     with open(test_script_stdout, 'w', encoding='utf-8') as stdout:
    #         subprocess.run(
    #             [python_executable, test_script, "-1", ROOT_DIR, "train", str(cfg.problem.case_num)], 
    #             stdout=stdout, stderr=subprocess.STDOUT, timeout=300
    #         )
    #     logging.info(f"Validation finished. Results saved to: {test_script_stdout}")
    # except subprocess.TimeoutExpired:
    #     logging.error("Validation script timed out")
    # except Exception as e:
    #     logging.error(f"Error running validation: {e}")


if __name__ == "__main__":
    main()
