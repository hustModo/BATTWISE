import os
from .original.eoh import EOH
from .original.getParas import Paras
from .original import prob_rank, pop_greedy
from .problem_adapter import Problem

from utils.utils import init_client

class EoH:
    def __init__(self, cfg, root_dir, case_num, output_dir=None) -> None:
        self.cfg = cfg
        self.root_dir = root_dir
        self.case_num = case_num
        
        # 设置输出目录
        self.output_dir = output_dir if output_dir else "./"
        self.problem = Problem(cfg, root_dir, case_num, output_dir=self.output_dir)

        self.paras = Paras() 
        self.paras.set_paras(method = "eoh",
                    llm_model = cfg.model,
                    ec_pop_size = cfg.pop_size,
                    ec_n_pop = cfg.max_fe,  # total evals = 2 * pop_size + n_pop * 4 * pop_size; for pop_size = 10, n_pop = 5, total evals = 2 * 10 + 4 * 5 * 10 = 220
                    exp_output_path = self.output_dir,
                    exp_debug_mode = False,
                    eva_timeout=cfg.timeout,
                    exp_use_seed=bool(getattr(cfg, 'exp_use_seed', False)),
                    exp_seed_path=str(getattr(cfg, 'exp_seed_path', './seeds/seeds.json')))
        init_client(self.cfg)
    
    def evolve(self):
        print("- Evolution Start -")

        # 传递fast_eval配置给EOH
        fast_eval = getattr(self.cfg, 'fast_eval', False)
        method = EOH(self.paras, self.problem, prob_rank, pop_greedy, fast_eval=fast_eval)

        results = method.run()

        print("> End of Evolution! ")
        print("----------------------------------------- ")
        print("---     EoH successfully finished !   ---")
        print("-----------------------------------------")

        return results


