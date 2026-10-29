import numpy as np
import json
import random
import time
import glob
import os
import re
import logging

from .eoh_interface_EC import InterfaceEC
# main class for eoh
class EOH:

    # initilization
    def __init__(self, paras, problem, select, manage, **kwargs):

        self.prob = problem
        self.select = select
        self.manage = manage
        
        # LLM settings
        self.use_local_llm = paras.llm_use_local
        self.url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint  # currently only API2D + GPT
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        # RZ: a quick fix for fast evaluation
        # 修改：从配置读取，默认使用LLM生成初始种群
        self.fast_eval = kwargs.get('fast_eval', False)
        # ------------------ RZ: use local LLM ------------------
        self.use_local_llm = kwargs.get('use_local_llm', False)
        assert isinstance(self.use_local_llm, bool)
        if self.use_local_llm:
            assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
            assert isinstance(kwargs.get('url'), str)
            self.url = kwargs.get('url')
        # -------------------------------------------------------

        # Experimental settings       
        self.pop_size = paras.ec_pop_size  # popopulation size, i.e., the number of algorithms in population
        self.n_pop = paras.ec_n_pop  # number of populations
        
        # Textgrad回调函数（用于反向传播）
        self.textgrad_callback = kwargs.get('textgrad_callback', None)
        self.generation_count = 0
        
        # Textgrad guidance provider（用于前向指导）
        self.textgrad_guidance_provider = kwargs.get('textgrad_guidance_provider', None)
        
        # LLM并发控制
        self.llm_max_workers = getattr(paras, 'llm_max_workers', kwargs.get('llm_max_workers', None))

        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        if paras.ec_m > self.pop_size or paras.ec_m == 1:
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.ec_m = 2
        self.m = paras.ec_m

        self.debug_mode = paras.exp_debug_mode  # if debug
        self.ndelay = 1  # default

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path

        self.exp_n_proc = paras.exp_n_proc
        
        self.timeout = paras.eva_timeout

        self.use_numba = paras.eva_numba_decorator

        print("- EoH parameters loaded -")

        # Set a random seed
        random.seed(2024)

    # add new individual to population
    def add2pop(self, population, offspring):
        for off in offspring:
            for ind in population:
                if ind['objective'] == off['objective']:
                    if (self.debug_mode):
                        print("duplicated result, retrying ... ")
            population.append(off)
    

    # run eoh 
    def run(self):

        print("- Evolution Start -")

        time_start = time.time()

        # interface for large language model (llm)
        # interface_llm = PromptLLMs(self.api_endpoint,self.api_key,self.llm_model,self.debug_mode)

        # interface for evaluation
        interface_prob = self.prob

        # interface for ec operators
        interface_ec = InterfaceEC(self.pop_size, self.m, self.api_endpoint, self.api_key, self.llm_model,
                                   self.debug_mode, interface_prob, use_local_llm=self.use_local_llm, url=self.url, select=self.select,n_p=self.exp_n_proc,
                                   timeout = self.timeout, use_numba=self.use_numba,
                                   guidance_provider=self.textgrad_guidance_provider,  # ✨ 传递guidance provider
                                   llm_max_workers=self.llm_max_workers  # ✨ 传递LLM并发控制
                                   )

        # initialization
        population = []
        if self.fast_eval:
            logging.info("Fast evaluation mode: loading initial population")
            
            try:
                # 构建路径：使用相对路径遻免硬编码
                current_file_path = os.path.dirname(os.path.abspath(__file__))
                # baselines/eoh/original -> project_root
                project_root = os.path.abspath(os.path.join(current_file_path, '..', '..', '..'))
                
                # 允许从配置中指定路径，否则使用默认值
                extracted_codes_dir = getattr(self.prob, 'fast_eval_path', None) or \
                                     os.path.join(project_root, 'outputs', 'extracted_codes')
                
                if not os.path.exists(extracted_codes_dir):
                    logging.warning(f"Fast eval directory not found: {extracted_codes_dir}")
                    logging.info("Creating empty initial population and falling back to normal initialization")
                    population = []
                    n_start = 0
                else:
                    logging.info(f"Loading from: {extracted_codes_dir}")
                    search_path = os.path.join(extracted_codes_dir, '*')
                    subdirs = [d for d in glob.glob(search_path) if os.path.isdir(d)]
                    
                    initial_data = []
                    for subdir in subdirs:
                        file_path = os.path.join(subdir, 'problem_iter0_response0.txt')
                        if os.path.exists(file_path):
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    # Extract algorithm description
                                    alg_match = re.search(r"\{(.*)\}", content, re.DOTALL)
                                    alg_desc = alg_match.group(1).strip() if alg_match else "Algorithm description not found"
                                    
                                    # Extract code
                                    code_match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
                                    if code_match:
                                        code = code_match.group(1).strip()
                                        initial_data.append({'algorithm': alg_desc, 'code': code})
                                        logging.debug(f"Loaded code from {file_path}")
                            except Exception as e:
                                logging.warning(f"Error reading {file_path}: {e}")
                                continue
                    
                    if not initial_data:
                        logging.warning("No initial data could be parsed from fast_eval directory")
                        logging.info("Falling back to empty initial population")
                        population = []
                    else:
                        logging.info(f"Loaded {len(initial_data)} code samples from fast_eval")
                        # Evaluate the initial population
                        codes = [d['code'] for d in initial_data]
                        try:
                            objectives = interface_ec.interface_eval.batch_evaluate(codes, 0)
                            
                            population = []
                            for i, data in enumerate(initial_data):
                                individual = {
                                    'algorithm': data['algorithm'],
                                    'code': data['code'],
                                    'objective': np.round(objectives[i], 5) if objectives[i] is not None else float('inf'),
                                    'other_inf': None
                                }
                                population.append(individual)
                            logging.info(f"Successfully evaluated {len(population)} individuals")
                        except Exception as e:
                            logging.error(f"Error evaluating initial population: {e}")
                            import traceback
                            logging.error(traceback.format_exc())
                            population = []
                    
                    population = self.manage.population_management(population, self.pop_size)
                    
                    # Log initial best solution (fast_eval mode)
                    if population:
                        best_initial = population[0]
                        logging.info(f"=" * 80)
                        logging.info(f"Fast Eval - Initial Best: {best_initial['objective']:.6f}")
                        logging.info(f"=" * 80)
                        print(f"🏆 Initial Best Objective: {best_initial['objective']:.6f}")
                    
                    n_start = 0
            
            except Exception as e:
                logging.error(f"Critical error in fast_eval mode: {e}")
                import traceback
                logging.error(traceback.format_exc())
                logging.info("Falling back to empty population")
                population = []
                n_start = 0

        elif self.use_seed:
            with open(self.seed_path) as file:
                data = json.load(file)
            population = interface_ec.population_generation_seed(data)
            filename = os.path.join(self.output_path, "population_generation_0.json")
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
            
            # Log initial best solution (seed mode)
            if population:
                best_initial = population[0]
                logging.info(f"=" * 80)
                logging.info(f"Generation 0 (Seed) - BEST SOLUTION:")
                logging.info(f"  Objective Value: {best_initial['objective']:.6f}")
                logging.info(f"  Algorithm Description: {best_initial.get('algorithm', 'N/A')[:200]}...")
                logging.info(f"=" * 80)
                print(f"🏆 Initial Best Objective: {best_initial['objective']:.6f}")
            
            n_start = 0
        else:
            if self.load_pop:  # load population from files
                print("load initial population from " + self.load_pop_path)
                with open(self.load_pop_path) as file:
                    data = json.load(file)
                for individual in data:
                    population.append(individual)
                print("initial population has been loaded!")
                n_start = self.load_pop_id
            else:  # create new population
                print("creating initial population:")
                population = interface_ec.population_generation()
                population = self.manage.population_management(population, self.pop_size)

                # print(len(population))
                # if len(population)<self.pop_size:
                #     for op in [self.operators[0],self.operators[2]]:
                #         _,new_ind = interface_ec.get_algorithm(population, op)
                #         self.add2pop(population, new_ind)
                #         population = self.manage.population_management(population, self.pop_size)
                #         if len(population) >= self.pop_size:
                #             break
                #         print(len(population))
     
                
                print(f"Pop initial: ")
                for off in population:
                    print(" Obj: ", off['objective'], end="|")
                print()
                print("initial population has been created!")
                # Save population to a file
                filename = os.path.join(self.output_path, "population_generation_0.json")
                with open(filename, 'w') as f:
                    json.dump(population, f, indent=5)
                
                # Log initial best solution
                if population:
                    best_initial = population[0]
                    logging.info(f"=" * 80)
                    logging.info(f"Generation 0 (Initial) - BEST SOLUTION:")
                    logging.info(f"  Objective Value: {best_initial['objective']:.6f}")
                    logging.info(f"  Algorithm Description: {best_initial.get('algorithm', 'N/A')[:200]}...")
                    logging.info(f"=" * 80)
                    print(f"🏆 Initial Best Objective: {best_initial['objective']:.6f}")
                
                n_start = 0

        # main loop
        if self.fast_eval:
            self.operators = ["i1", "e1", "m1"]
            self.n_pop = 1
            
        n_op = len(self.operators)
        
        # 创建最优解演化轨迹文件
        best_solutions_log = os.path.join(self.output_path, "best_solutions_evolution.txt")
        with open(best_solutions_log, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(f"{'Generation':^15} | {'Best Objective':^20} | {'Improvement':^20}\n")
            f.write("=" * 100 + "\n")
        
        # 计算每个算子应该生成多少个个体（平均分配）
        # 例如：pop_size=20, 4个算子 -> 每个算子生成5个
        offspring_per_op = self.pop_size // n_op
        remainder = self.pop_size % n_op  # 处理不能整除的情况
        
        index_pop = 1
        for pop in range(n_start, self.n_pop):
            self.generation_count = pop + 1  # 更新代数计数器
            print(f"\n=== Generation {pop + 1}/{self.n_pop} ===")
            
            # 🚀 批量模式：收集所有算子的提示词，一起LLM调用和评估
            all_operator_info = []  # 存储每个算子的信息
            
            for i in range(n_op):
                op = self.operators[i]
                n_offspring = offspring_per_op + (1 if i < remainder else 0)
                op_w = self.operator_weights[i]
                
                if np.random.rand() < op_w:
                    all_operator_info.append({
                        'operator': op,
                        'n_offspring': n_offspring,
                        'index': i
                    })
                    print(f" OP: {op}(×{n_offspring})", end=" | ")
                else:
                    print(f" OP: {op}(×{n_offspring}) Skipped", end=" | ")
            
            print()  # 换行
            
            # 一次性批量生成和评估所有算子的个体
            if all_operator_info:
                parents_list, offsprings = interface_ec.get_algorithm_batch(
                    population, all_operator_info, index_pop
                )
                
                # 添加到种群
                self.add2pop(population, offsprings)
                
                # 🔄 Textgrad反向传播回调
                if self.textgrad_callback and offsprings:
                    for i, offspring in enumerate(offsprings):
                        # 获取该offspring对应的parents
                        parents = parents_list[i] if i < len(parents_list) else []
                        operator_used = offspring.get('operator', 'unknown')
                        
                        # 调用textgrad回调
                        try:
                            self.textgrad_callback(
                                offspring=offspring,
                                parents=parents,
                                operator_type=operator_used,
                                generation=self.generation_count
                            )
                        except Exception as e:
                            logging.warning(f"Textgrad callback error: {e}")
                
                # 打印结果
                print(f"Generated {len(offsprings)} offspring: ", end="")
                for off in offsprings:
                    print(f"{off['objective']:.2f}", end=" | ")
                print()
                
                # 种群管理
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
            
            index_pop += 1


            # Save population to a file
            filename = os.path.join(self.output_path, f"population_generation_{pop + 1}.json")
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)

            # Save the best one to a file
            filename = os.path.join(self.output_path, f"best_population_generation_{pop + 1}.json")
            with open(filename, 'w') as f:
                json.dump(population[0], f, indent=5)

            # Log best solution information
            best_individual = population[0]
            logging.info(f"=" * 80)
            logging.info(f"Generation {pop + 1}/{self.n_pop} - BEST SOLUTION:")
            logging.info(f"  Objective Value: {best_individual['objective']:.6f}")
            logging.info(f"  Algorithm Description: {best_individual.get('algorithm', 'N/A')[:200]}...")
            logging.info(f"  Saved to: {filename}")
            logging.info(f"=" * 80)
            
            # 记录到演化轨迹文件
            with open(best_solutions_log, 'a', encoding='utf-8') as f:
                if pop == 0:
                    improvement = "Initial"
                    f.write(f"{f'Gen {pop + 1}':^15} | {best_individual['objective']:^20.6f} | {improvement:^20}\n")
                else:
                    # 读取上一代的最优值来计算改进
                    prev_best_file = os.path.join(self.output_path, f"best_population_generation_{pop}.json")
                    if os.path.exists(prev_best_file):
                        with open(prev_best_file, 'r') as pf:
                            prev_best = json.load(pf)
                            prev_obj = prev_best['objective']
                            improvement = prev_obj - best_individual['objective']
                            improvement_pct = (improvement / prev_obj * 100) if prev_obj != 0 else 0
                            improvement_str = f"{improvement:+.2f} ({improvement_pct:+.2f}%)"
                    else:
                        improvement_str = "N/A"
                    f.write(f"{f'Gen {pop + 1}':^15} | {best_individual['objective']:^20.6f} | {improvement_str:^20}\n")

            print(f"--- {pop + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time()-time_start)/60):.1f} m")
            print(f"🏆 Best Objective: {best_individual['objective']:.6f}")
            print("Pop Objs: ", end=" ")
            for i in range(len(population)):
                print(str(population[i]['objective']) + " ", end="")
            print()
        
        return population[0]["code"], filename, population[0]["objective"]


