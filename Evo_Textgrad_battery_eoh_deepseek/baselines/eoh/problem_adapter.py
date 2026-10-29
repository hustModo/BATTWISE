import logging
import os
import subprocess
import re
import numpy as np
from multiprocessing import Pool, Manager
import time

from utils.utils import block_until_running, file_to_string, filter_traceback

class Prompts:
    def __init__(self, problem_cfg, root_dir:str):
        self.cfg = problem_cfg
        self.problem = problem_cfg.problem_name
        self.root_dir = root_dir
        self.problem_type = problem_cfg.problem_type
        self.prompt_dir = f"{self.root_dir}/prompts"

        prompt_path_suffix = "_black_box" if self.problem_type == "black_box" else ""
        problem_prompt_path = f'{self.prompt_dir}/{self.problem}{prompt_path_suffix}'

        self.seed_func = file_to_string(f'{problem_prompt_path}/seed_func.txt')
        self.func_signature = file_to_string(f'{problem_prompt_path}/func_signature.txt').format(version=2).strip()
        self.func_desc = file_to_string(f'{problem_prompt_path}/func_desc.txt')

        match = re.match(r'^def +(.+?_v2)\((.*)\) *-> *(.*?) *:', self.func_signature)
        assert match is not None
        self.prompt_func_name = match.group(1)
        # 解析函数参数，处理复杂的参数列表
        params_str = match.group(2)
        self.prompt_func_inputs = []
        for param in params_str.split(','):
            param = param.strip()
            if ':' in param:
                param_name = param.split(':')[0].strip()
                self.prompt_func_inputs.append(param_name)
            else:
                self.prompt_func_inputs.append(param)
        
        if self.prompt_func_name.startswith('select_next_node'):
            self.prompt_func_outputs = ['next_node']
        elif self.prompt_func_name.startswith('priority'):
            self.prompt_func_outputs = ['priority']
        elif self.prompt_func_name.startswith('heuristics'):
            self.prompt_func_outputs = ['heuristics_matrix']
        elif self.prompt_func_name.startswith('crossover'):
            self.prompt_func_outputs = ['offsprings']
        elif self.prompt_func_name.startswith('get_combined_expression'):
            self.prompt_func_outputs = ['combined_expression_data']
        elif self.prompt_func_name.startswith('calculate_mmd'):
            self.prompt_func_outputs = ['mmd_loss']
        else:
            self.prompt_func_outputs = ['result']

    def get_task(self):
        return self.cfg.description
    
    def get_func_name(self):
        return self.prompt_func_name
    
    def get_func_inputs(self):
        return self.prompt_func_inputs
    
    def get_func_outputs(self):
        return self.prompt_func_outputs
    
    def get_inout_inf(self):
        return self.func_desc

    def get_other_inf(self):
        return ""

class Problem:
    def __init__(self, cfg, root_dir, case_num, output_dir=None):
        self.config = cfg
        self.root_dir = root_dir
        self.case_num = case_num
        
        # 设置输出目录
        self.output_dir = output_dir if output_dir else "./"
        # evaluations 作为子目录
        self.eval_dir = os.path.join(self.output_dir, 'evaluations')
        if not os.path.isdir(self.eval_dir):
            os.makedirs(self.eval_dir, exist_ok=True)

        self.problem = self.config.problem.problem_name
        self.problem_description = self.config.problem.description
        self.problem_size = self.config.problem.problem_size
        self.obj_type = self.config.problem.obj_type
        self.problem_type = self.config.problem.problem_type
        self.output_file = f"{self.root_dir}/problems/{self.problem}/gpt.py"

        if self.problem_type == "tsp_constructive":
            from .original.prompts.tsp_greedy import GetPrompts
            self.prompts = GetPrompts()
        elif self.problem_type == "bpp_online":
            from .original.prompts.bpp_online import GetPrompts
            self.prompts = GetPrompts()
        else:
            self.prompts = Prompts(self.config.problem, root_dir)

    def response_to_individual(self, code, response_id, file_name=None) -> dict:
        """
        Convert response to individual
        """
        # 使用统一的evaluations目录
        outdir = self.eval_dir + '/'
        runid = hash(code)
        # Write response to file
        file_name = outdir+f"problem_eval{runid}.txt" if file_name is None else file_name + ".txt"
        with open(file_name, 'w', encoding='utf-8') as file:
            file.writelines(code + '\n')

        # Extract code and description from response
        std_out_filepath = outdir+f"problem_eval{runid}_stdout.txt" if file_name is None else file_name.rstrip(".txt") + "_stdout.txt"
        
        individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": outdir+f"problem_eval{runid}_code.py",
            "code": code,
            "response_id": response_id,
        }
        return individual
    
    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        """
        Mark an individual as invalid.
        """
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["traceback_msg"] = traceback_msg
        return individual

    def batch_evaluate(self, codes: list[str], iteration: int) -> list[dict]:
        population = [self.response_to_individual(code, i) for i, code in enumerate(codes)]
        population = self.evaluate_population(population, iteration)
        return [indiv["obj"] for indiv in population]

    def evaluate_population(self, population: list[dict], iteration: int) -> list[dict]:
        """
        Evaluate population by running code in parallel, with each process using a unique gpt_code_X.py file.
        """
        processes = []
        # Create a dedicated directory for generated code files (eval.py expects this path)
        gpt_file_dir = os.path.join(self.root_dir, 'baselines', 'eoh', 'gpt_code')
        os.makedirs(gpt_file_dir, exist_ok=True)

        # First loop: Start all evaluation subprocesses
        for response_id, individual in enumerate(population):
            if individual["code"] is None:
                population[response_id] = self.mark_invalid_individual(individual, "Invalid response!")
                processes.append(None)
                continue

            logging.info(f"Iteration {iteration}: Starting evaluation for Code {response_id}")
            try:
                # Write code to its own unique gpt_code_X.py file
                # 每次评估都为每个response_id创建独立的文件，避免并发冲突
                gpt_file_path = os.path.join(gpt_file_dir, f'gpt_code_{response_id}.py')
                with open(gpt_file_path, 'w', encoding='utf-8') as file:
                    file.write(individual["code"])
                    file.flush()
                    os.fsync(file.fileno())

                process = self._run_code(individual, response_id, iteration, self.case_num)
                processes.append(process)
            except Exception as e:
                logging.info(f"Error starting process for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(individual, str(e))
                processes.append(None)

        # Second loop: Wait for processes and collect results in the same order
        for response_id, process in enumerate(processes):
            if process is None:
                continue

            individual = population[response_id]
            try:
                process.communicate(timeout=self.config.timeout)
            except subprocess.TimeoutExpired as e:
                logging.info(f"Timeout Error for response_id {response_id}: {e}")
                process.kill()
                population[response_id] = self.mark_invalid_individual(individual, str(e))
                continue

            # Process the result from the stdout file
            stdout_filepath = individual["stdout_filepath"]
            try:
                with open(stdout_filepath, 'r', encoding='utf-8') as f:
                    stdout_str = f.read()
                
                traceback_msg = filter_traceback(stdout_str)
                if traceback_msg == '':
                    try:
                        # 优先解析"Error code:"格式（用于battery_mmd等新问题）
                        if "Error code:" in stdout_str:
                            for line in reversed(stdout_str.split('\n')):
                                if line.strip().startswith("Error code:"):
                                    obj_str = line.split(":")[-1].strip()
                                    obj = float(obj_str)
                                    break
                            else:
                                raise ValueError("Error code line found but could not parse")
                        else:
                            # 传统格式：倒数第二行是objective
                            obj = float(stdout_str.split('\n')[-2])
                        
                        # 处理objective值
                        if obj == float('inf') or obj == float('-inf'):
                            individual["obj"] = float('inf')
                            individual["exec_success"] = False
                        else:
                            # 根据obj_type调整objective
                            if self.obj_type == "max":
                                # 最大化问题：eval.py应该返回负值（因为EOH最小化）
                                # 如果返回的已经是负值，直接使用；如果是正值，取负
                                individual["obj"] = obj if obj < 0 else -obj
                            else:
                                # 最小化问题：直接使用正值
                                individual["obj"] = abs(obj)
                            individual["exec_success"] = True
                    except Exception as e:
                        population[response_id] = self.mark_invalid_individual(individual, f"Invalid std out / objective value! {e}")
                else:
                    population[response_id] = self.mark_invalid_individual(individual, traceback_msg)
            except FileNotFoundError:
                 population[response_id] = self.mark_invalid_individual(individual, f"Stdout file not found: {stdout_filepath}")


            logging.info(f"Iteration {iteration}, response_id {response_id}: Objective value: {individual.get('obj', 'N/A')}")

        return population

    def _run_code(self, individual: dict, response_id: int, iteration: int, case_num: list) -> subprocess.Popen:
        """
        Run eval script, passing the response_id as the island_id.
        """
        logging.debug(f"Iteration {iteration}: Processing Code Run {response_id}")

        # Execute the python file with flags
        # 确保case_num正确转换为字符串
        if isinstance(case_num, list):
            case_num_str = ' '.join(str(num) for num in case_num)
        else:
            case_num_str = str(case_num)
        cell_filter = getattr(self.config, "eoh_cell_filter", None)
        if cell_filter is None:
            cell_filter_arg = ""
        elif isinstance(cell_filter, (list, tuple)):
            cell_filter_arg = ",".join(str(int(c)) for c in cell_filter)
        else:
            cell_filter_arg = str(cell_filter)
        
        with open(individual["stdout_filepath"], 'w', encoding='utf-8') as f:
            eval_file_path = f'{self.root_dir}/problems/{self.problem}/eval.py' if self.problem_type != "black_box" else f'{self.root_dir}/problems/{self.problem}/eval_black_box.py'
            
            # Pass the response_id as the island_id to the evaluation script
            command = ['python', '-u', eval_file_path, self.root_dir, "train", case_num_str, str(response_id)]
            if cell_filter_arg:
                command.append(cell_filter_arg)
            process = subprocess.Popen(command, stdout=f, stderr=f)

        block_until_running(individual["stdout_filepath"], log_status=True, iter_num=iteration, response_id=response_id)
        return process
