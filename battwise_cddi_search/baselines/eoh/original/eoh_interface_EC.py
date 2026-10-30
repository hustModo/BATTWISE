import numpy as np
import time
from .eoh_evolution import Evolution
import warnings
from joblib import Parallel, delayed
import re
import concurrent.futures

class InterfaceEC():
    def __init__(self, pop_size, m, api_endpoint, api_key, llm_model, debug_mode, interface_prob, select,n_p,timeout,use_numba,**kwargs):
        # -------------------- RZ: use local LLM --------------------
        assert 'use_local_llm' in kwargs
        assert 'url' in kwargs
        # -----------------------------------------------------------

        # LLM settings
        self.pop_size = pop_size
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        # Textgrad guidance provider
        self.guidance_provider = kwargs.get('guidance_provider', None)
        self.evol = Evolution(api_endpoint, api_key, llm_model, debug_mode,prompts, **kwargs)
        self.m = m
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select
        self.n_p = n_p
        
        self.timeout = timeout
        self.use_numba = use_numba
        
    def evaluate_population_from_codes(self, codes):
        objectives = self.interface_eval.batch_evaluate(codes, 0)
        population = []
        for i, code in enumerate(codes):
            individual = {
                'algorithm': None,
                'code': code,
                'objective': np.round(objectives[i], 5) if objectives[i] is not None else float('inf'),
                'other_inf': None
            }
            population.append(individual)
        return population

    def code2file(self,code):
        with open("./ael_alg.py", "w") as file:
        # Write the code to the file
            file.write(code)
        return 
    
    def add2pop(self,population,offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True
    
    def check_duplicate(self,population,code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    # def population_management(self,pop):
    #     # Delete the worst individual
    #     pop_new = heapq.nsmallest(self.pop_size, pop, key=lambda x: x['objective'])
    #     return pop_new
    
    # def parent_selection(self,pop,m):
    #     ranks = [i for i in range(len(pop))]
    #     probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    #     parents = random.choices(pop, weights=probs, k=m)
    #     return parents

    def population_generation(self):
        
        _, population = self.get_algorithm([], 'i1', 0)
             
        return population
    
    def population_generation_seed(self,seeds):

        population = []

        fitness = self.interface_eval.batch_evaluate([seed['code'] for seed in seeds])

        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None,
                    'other_inf': None
                }

                obj = np.array(fitness[i])
                seed_alg['objective'] = np.round(obj, 5)
                population.append(seed_alg)

            except Exception as e:
                print("Error in seed algorithm")
                exit()

        print("Initiliazation finished! Get "+str(len(seeds))+" seed algorithms")

        return population
    

    def _get_alg(self,pop,operator):
        offspring = {
            'algorithm': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        if operator == "i1":
            parents = None
            [offspring['code'],offspring['algorithm']] =  self.evol.i1()            
        elif operator == "e1":
            parents = self.select.parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']] = self.evol.e1(parents)
        elif operator == "e2":
            parents = self.select.parent_selection(pop,self.m)
            [offspring['code'],offspring['algorithm']] = self.evol.e2(parents) 
        elif operator == "m1":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.m1(parents[0])   
        elif operator == "m2":
            parents = self.select.parent_selection(pop,1)
            [offspring['code'],offspring['algorithm']] = self.evol.m2(parents[0]) 
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n") 

        return parents, offspring

    def get_algorithm(self, pop, operator, index_pop, n_offspring=None):
        """
        Generate algorithms using the specified operator.
        
        Args:
            pop: Current population
            operator: Operator type (i1, e1, e2, m1, m2)
            index_pop: Population index
            n_offspring: Number of offspring to generate. If None, uses self.pop_size
        """
        prompts = []
        parents_list = []
        
        # 确定生成个体数量
        num_to_generate = n_offspring if n_offspring is not None else self.pop_size

        # 1. Collect prompts
        for _ in range(num_to_generate):
            if operator == "i1":
                parents = None
                prompts.append(self.evol.get_prompt_i1())
            elif operator in ["e1", "e2"]:
                parents = self.select.parent_selection(pop, self.m)
                if operator == "e1":
                    prompts.append(self.evol.get_prompt_e1(parents))
                else: # e2
                    prompts.append(self.evol.get_prompt_e2(parents))
            elif operator in ["m1", "m2"]:
                parents = self.select.parent_selection(pop, 1)
                if operator == "m1":
                    prompts.append(self.evol.get_prompt_m1(parents[0]))
                else: # m2
                    prompts.append(self.evol.get_prompt_m2(parents[0]))
            else:
                 print(f"Evolution operator [{operator}] has not been implemented ! \n")
                 continue
            parents_list.append(parents)

        # 2. Batch inference
        results = self.evol._get_alg_batch(prompts)

        # 3. Batch evaluation
        codes = [res[0] for res in results if res[0] is not None]
        if not codes:
            return [], []
            
        objectives = self.interface_eval.batch_evaluate(codes, index_pop)

        # 4. Construct offsprings
        offsprings = []
        code_idx = 0
        for i, (code, algorithm) in enumerate(results):
            if code is None:
                continue

            offspring = {
                'algorithm': algorithm,
                'code': code,
                'objective': np.round(objectives[code_idx], 5),
                'other_inf': None
            }
            offsprings.append(offspring)
            code_idx += 1

        return parents_list, offsprings
    
    def get_algorithm_batch(self, pop, operator_info_list, index_pop):
        """
        批量生成多个算子的个体（提高并行度）
        
        Args:
            pop: Current population
            operator_info_list: List of dicts with keys: 'operator', 'n_offspring', 'index'
            index_pop: Population index
        
        Returns:
            parents_list: List of parents (for compatibility)
            offsprings: List of all generated offspring
        """
        all_prompts = []
        all_parents_list = []
        operator_boundaries = []  # 记录每个算子的边界
        
        # 1. 收集所有算子的提示词
        current_idx = 0
        for op_info in operator_info_list:
            operator = op_info['operator']
            n_offspring = op_info['n_offspring']
            
            start_idx = current_idx
            for _ in range(n_offspring):
                if operator == "i1":
                    parents = None
                    all_prompts.append(self.evol.get_prompt_i1())
                elif operator in ["e1", "e2"]:
                    parents = self.select.parent_selection(pop, self.m)
                    if operator == "e1":
                        all_prompts.append(self.evol.get_prompt_e1(parents))
                    else:
                        all_prompts.append(self.evol.get_prompt_e2(parents))
                elif operator in ["m1", "m2"]:
                    parents = self.select.parent_selection(pop, 1)
                    if operator == "m1":
                        all_prompts.append(self.evol.get_prompt_m1(parents[0]))
                    else:
                        all_prompts.append(self.evol.get_prompt_m2(parents[0]))
                else:
                    print(f"Evolution operator [{operator}] has not been implemented ! \n")
                    continue
                all_parents_list.append(parents)
                current_idx += 1
            
            end_idx = current_idx
            operator_boundaries.append({
                'operator': operator,
                'start': start_idx,
                'end': end_idx
            })
        
        if not all_prompts:
            return [], []
        
        # 2. 批量LLM推理（所有算子一起）
        print(f"Batch LLM inference: {len(all_prompts)} prompts")
        results = self.evol._get_alg_batch(all_prompts)
        
        # 3. 批量评估（所有算子一起）
        codes = [res[0] for res in results if res[0] is not None]
        if not codes:
            return [], []
        
        print(f"Batch evaluation: {len(codes)} codes")
        objectives = self.interface_eval.batch_evaluate(codes, index_pop)
        
        # 4. 构建所有offspring，并添加算子类型标记
        offsprings = []
        code_idx = 0
        for i, (code, algorithm) in enumerate(results):
            if code is None:
                continue
            
            # 找到该offspring对应的算子
            operator_type = 'unknown'
            for boundary in operator_boundaries:
                if boundary['start'] <= i < boundary['end']:
                    operator_type = boundary['operator']
                    break
            
            offspring = {
                'algorithm': algorithm,
                'code': code,
                'objective': np.round(objectives[code_idx], 5),
                'other_inf': None,
                'operator': operator_type  # ✨ 添加算子类型标记
            }
            offsprings.append(offspring)
            code_idx += 1
        
        return all_parents_list, offsprings