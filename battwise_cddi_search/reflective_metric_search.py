"""
BATTWISE performance-guided reflective metric discovery.

Core mechanism:
1. 使用EOH的多样化算子（i1, e1, e2, m1, m2）进行启发式生成
2. Generate evidence-based reflection from measured candidate performance
3. Maintain hierarchical short-term, consolidated, and operator-specific memory
4. Adapt proposal guidance and operator weights from observed gains
"""

import numpy as np
import logging
import subprocess
import os
import copy
import json
import time
from typing import List, Dict, Tuple, Optional
from utils.utils import *

# EOH相关导入
from baselines.eoh.eoh_adapter import EoH
from baselines.eoh.original.eoh import EOH
from baselines.eoh.original.getParas import Paras
from baselines.eoh.original import prob_rank, pop_greedy
from baselines.eoh.problem_adapter import Problem

import sys
python_executable = sys.executable


class HierarchicalReflectionMemory:
    """反思库：存储和管理优化知识"""
    
    def __init__(self, max_short_term: int = 20):
        self.short_term_reflections = []  # 短期反思队列
        self.long_term_reflection = ""    # 长期反思
        self.operator_specific_guidance = {  # 算子特定指导
            'i1': [],
            'e1': [],
            'e2': [],
            'm1': [],
            'm2': []
        }
        self.max_short_term = max_short_term
        self.performance_history = []  # 性能历史
        
    def add_reflection(self, reflection: str, operator_type: str, 
                      performance_gain: float, timestamp: float = None):
        """添加新的短期反思"""
        if timestamp is None:
            timestamp = time.time()
            
        self.short_term_reflections.append({
            'content': reflection,
            'operator': operator_type,
            'gain': performance_gain,
            'timestamp': timestamp
        })
        
        # 保留最近N个反思
        if len(self.short_term_reflections) > self.max_short_term:
            self.short_term_reflections.pop(0)
    
    def update_operator_guidance(self, operator_type: str, guidance: str):
        """更新算子特定的优化指导"""
        self.operator_specific_guidance[operator_type].append(guidance)
        # 保留最近3个指导
        if len(self.operator_specific_guidance[operator_type]) > 3:
            self.operator_specific_guidance[operator_type].pop(0)
    
    def get_guidance_for_operator(self, operator_type: str) -> str:
        """获取特定算子的优化指导（带长度控制）"""
        # 结合长期反思和算子特定指导
        guidance_parts = []
        
        if self.long_term_reflection:
            guidance_parts.append("[Long-term Optimization Guidance]")
            guidance_parts.append(self.long_term_reflection)
        
        if self.operator_specific_guidance[operator_type]:
            guidance_parts.append(f"\n[{operator_type.upper()} Operator-Specific Insights]")
            # 只使用最近3条算子特定指导
            recent_guidance = self.operator_specific_guidance[operator_type][-3:]
            guidance_parts.extend(recent_guidance)
        
        guidance = "\n".join(guidance_parts) if guidance_parts else ""
        
        # ✨ 监控长度
        if guidance:
            word_count = len(guidance.split())
            char_count = len(guidance)
            if word_count > 300:
                logging.warning(f"[ReflectiveSearch] Guidance for {operator_type} is long: {word_count} words, {char_count} chars")
            else:
                logging.debug(f"[ReflectiveSearch] Guidance for {operator_type}: {word_count} words, {char_count} chars")
        
        return guidance
    
    def get_recent_reflections(self, n: int = 10) -> List[Dict]:
        """获取最近N个反思"""
        return self.short_term_reflections[-n:]
    
    def calculate_operator_performance(self) -> Dict[str, float]:
        """计算各算子的平均性能增益"""
        operator_gains = {'i1': [], 'e1': [], 'e2': [], 'm1': [], 'm2': []}
        
        for reflection in self.short_term_reflections:
            op = reflection['operator']
            gain = reflection['gain']
            if op in operator_gains:
                operator_gains[op].append(gain)
        
        # 计算平均值
        operator_scores = {}
        for op, gains in operator_gains.items():
            if gains:
                operator_scores[op] = np.mean(gains)
            else:
                operator_scores[op] = 0.0
        
        return operator_scores


class ReflectiveMetricSearch:
    """Performance-guided search for internal-state-aware battery metrics."""
    
    def __init__(self, cfg, root_dir, case_num, output_dir=None) -> None:
        self.cfg = cfg
        self.root_dir = root_dir
        self.case_num = case_num
        
        # 设置输出目录（使用Hydra的工作目录）
        self.output_dir = output_dir if output_dir else "./"
        
        # 初始化EOH组件
        self.problem = Problem(cfg, root_dir, case_num, output_dir=self.output_dir)
        self.paras = Paras()
        
        # 从配置获取算子列表
        eoh_operators = getattr(cfg, 'eoh_operators', ['e1', 'e2', 'm1', 'm2'])
        operator_weights_dict = getattr(cfg, 'operator_weights', {})
        
        # 转换权重为列表（与算子顺序对应）
        operator_weights = [operator_weights_dict.get(op, 1.0) for op in eoh_operators]
        
        # Seed support: when cfg.exp_use_seed is true, EOH will load
        # cfg.exp_seed_path (a JSON list of {algorithm, code}) as gen-0 instead
        # of generating from scratch. Used by BATTWISE to bootstrap from
        # historical winners so a small `max_fe` still produces strong results.
        self.paras.set_paras(
            method="reflective_metric_search",
            llm_model=cfg.model,
            ec_pop_size=cfg.pop_size,
            ec_n_pop=cfg.max_fe,
            ec_operators=eoh_operators,
            ec_operator_weights=operator_weights,
            exp_output_path=self.output_dir,
            exp_debug_mode=False,
            eva_timeout=cfg.timeout,
            llm_max_workers=getattr(cfg, 'llm_max_workers', None),
            exp_use_seed=bool(getattr(cfg, 'exp_use_seed', False)),
            exp_seed_path=str(getattr(cfg, 'exp_seed_path', './seeds/seeds.json')),
        )
        
        # 初始化客户端
        init_client(self.cfg)
        
        # Reflection configuration.  The fallback keeps historical YAML files
        # runnable while the canonical option is ``reflection_enabled``.
        self.reflection_enabled = getattr(
            cfg,
            'reflection_enabled',
            getattr(cfg, 'textgrad_enabled', True),
        )
        self.reflection_frequency = getattr(cfg, 'reflection_frequency', 5)
        self.adaptive_operator_selection = getattr(cfg, 'adaptive_operator_selection', True)
        self.performance_weighted = getattr(cfg, 'performance_weighted', True)
        
        # 初始化反思库
        max_reflections = getattr(cfg, 'max_short_term_reflections', 20)
        self.reflection_lib = HierarchicalReflectionMemory(max_short_term=max_reflections)
        
        # 算子权重（如果配置中未指定，使用默认值）
        default_weights = {'i1': 0.3, 'e1': 0.3, 'e2': 0.0, 'm1': 0.2, 'm2': 0.2}
        self.operator_weights = getattr(cfg, 'operator_weights', default_weights)
        
        # 迭代计数
        self.iteration = 0
        self.best_obj_overall = float('inf')
        self.best_code_overall = None
        
        # 初始化提示词
        self.init_prompts()
        
        logging.info("BATTWISE Reflective Metric Search initialized successfully")
        logging.info(f"Hierarchical reflection enabled: {self.reflection_enabled}")
        logging.info(f"Reflection frequency: {self.reflection_frequency}")
        logging.info(f"Adaptive operator selection: {self.adaptive_operator_selection}")
    
    def init_prompts(self):
        """初始化提示词"""
        self.prompt_dir = f"{self.root_dir}/prompts"
        problem_name = self.cfg.problem.problem_name
        
        # 加载EOH基础提示词（如果存在）
        eoh_prompt_path = f'{self.prompt_dir}/reflective_metric_search'
        if not os.path.exists(eoh_prompt_path):
            logging.warning(f"Reflective-search prompt directory not found: {eoh_prompt_path}")
            logging.warning("Will create default prompts")
            os.makedirs(eoh_prompt_path, exist_ok=True)
        
        # Load or create the reflective-search prompts.
        try:
            self.system_reflector_prompt = file_to_string(f'{self.prompt_dir}/reevo/system_reflector.txt')
        except:
            self.system_reflector_prompt = "You are an expert in algorithm optimization and reflection."
        
        self.performance_reflection_template = self._load_or_create_prompt(
            f'{eoh_prompt_path}/performance_reflection_stage1.txt',
            self._default_performance_reflection_prompt()
        )
        
        self.principle_synthesis_template = self._load_or_create_prompt(
            f'{eoh_prompt_path}/principle_synthesis_stage2.txt',
            self._default_principle_synthesis_prompt()
        )
        
        self.loss_calculate_template = self._load_or_create_prompt(
            f'{eoh_prompt_path}/loss_calculate.txt',
            self._default_loss_calculate_prompt()
        )
    
    def _load_or_create_prompt(self, path: str, default_content: str) -> str:
        """加载提示词文件，如果不存在则创建默认内容"""
        if os.path.exists(path):
            return file_to_string(path)
        else:
            # 创建默认提示词文件
            with open(path, 'w', encoding='utf-8') as f:
                f.write(default_content)
            return default_content
    
    def _default_performance_reflection_prompt(self) -> str:
        """Default prompt for candidate-performance reflection."""
        return """Analyze the performance of generated heuristic algorithm:

[Problem Description]
{problem_desc}

[Generated Algorithm]
{offspring_algorithm}
{offspring_code}

[Parent Algorithm 1]
{parent1_algorithm}
{parent1_code}

[Performance Ranking] (Lower is better)
1. {best_name}: {best_obj}
2. {middle_name}: {middle_obj}
3. {worst_name}: {worst_obj}

[Prior Optimization Guidance]
{prior_guidance}

**Analysis Task:**
Based on the performance comparison, analyze:
1. What makes the best-performing algorithm effective?
2. What are the weaknesses of the worst-performing algorithm?
3. How effective was the prior optimization guidance?
4. What specific improvements should be made next?

**Output Format:**
Provide optimization guidance in two parts:
1. **Core Principle** (5-8 words): The fundamental optimization strategy
2. **Specific Guidance** (20-30 words): Actionable implementation advice

Example: "Balance exploration and exploitation | Use adaptive parameters that adjust based on solution quality and search progress for better convergence."
"""
    
    def _default_principle_synthesis_prompt(self) -> str:
        """Default prompt for consolidating reflection into principles."""
        return """Synthesize optimization insights from recent algorithm evolution:

[Recent Short-term Reflections]
{short_term_reflections}

**Task:**
Extract the most valuable optimization principles from the reflections above.
Focus on patterns that led to consistent performance improvements.

**Output Format:**
Provide a concise optimization principle (one sentence, 15-25 words) that captures the essence of successful strategies.

Example: "Prioritize algorithms that balance multiple objectives while maintaining computational efficiency through adaptive parameter tuning."
"""
    
    def _default_loss_calculate_prompt(self) -> str:
        """默认的损失计算提示词（固定长度输出）"""
        return """Synthesize long-term optimization guidance:

[Recent High-Performance Reflections]
{new_reflections}

**CRITICAL Task:**
Extract EXACTLY 3 MOST IMPORTANT optimization principles from the reflections above.
Focus ONLY on what works best NOW.

**STRICT Requirements:**
1. Output EXACTLY 3 principles (no more, no less)
2. Each principle: ONE sentence, 10-20 words MAXIMUM
3. Focus on ACTIONABLE strategies
4. NO examples, NO explanations, NO elaborations
5. Be concise and specific

**Output Format:**
Principle 1: [10-20 words about the most effective strategy]
Principle 2: [10-20 words about the second most effective strategy]
Principle 3: [10-20 words about the third most effective strategy]

**Example Output:**
Principle 1: Adaptive bandwidth using median heuristic improves MMD sensitivity to distribution shifts.
Principle 2: Multi-scale kernels capture both local and global distribution differences effectively.
Principle 3: Normalized features prevent scale dominance in distance calculations.

[Prior Long-term Reflection - FOR REFERENCE ONLY]
{prior_reflection}
"""
    
    def build_performance_reflection_prompt(
        self,
        offspring: Dict,
        parents: List[Dict],
        performance_ranking: List[Tuple[str, float]],
    ) -> str:
        """Build a prompt grounded in measured candidate performance."""
        # 确定最优、中等、最差的算法
        best_name, best_obj = performance_ranking[0]
        middle_name, middle_obj = performance_ranking[1] if len(performance_ranking) > 1 else ("N/A", "N/A")
        worst_name, worst_obj = performance_ranking[2] if len(performance_ranking) > 2 else ("N/A", "N/A")
        
        # 获取对应的代码
        codes_map = {
            'Offspring': (offspring.get('algorithm', ''), offspring.get('code', '')),
            'Parent1': (parents[0].get('algorithm', ''), parents[0].get('code', '')) if len(parents) > 0 else ('', ''),
            'Parent2': (parents[1].get('algorithm', ''), parents[1].get('code', '')) if len(parents) > 1 else ('', '')
        }
        
        offspring_alg, offspring_code = codes_map['Offspring']
        parent1_alg, parent1_code = codes_map.get('Parent1', ('', ''))
        
        problem_desc = self.cfg.problem.description
        prior_guidance = self.reflection_lib.long_term_reflection or "No prior guidance available."
        
        prompt = self.performance_reflection_template.format(
            problem_desc=problem_desc,
            offspring_algorithm=offspring_alg,
            offspring_code=offspring_code,
            parent1_algorithm=parent1_alg,
            parent1_code=parent1_code,
            best_name=best_name,
            best_obj=best_obj,
            middle_name=middle_name,
            middle_obj=middle_obj,
            worst_name=worst_name,
            worst_obj=worst_obj,
            prior_guidance=prior_guidance
        )
        
        return prompt
    
    def generate_performance_reflection(self, offspring: Dict, parents: List[Dict],
                                        operator_type: str) -> Tuple[str, Optional[str]]:
        """Generate gain-aware reflection and consolidated search guidance.
        
        Returns:
            (short_term_reflection, core_principle)
        """
        if not self.reflection_enabled:
            return "", None
        
        # 计算性能排序
        individuals = [
            ('Offspring', offspring.get('objective', float('inf'))),
            ('Parent1', parents[0].get('objective', float('inf')) if len(parents) > 0 else float('inf')),
            ('Parent2', parents[1].get('objective', float('inf')) if len(parents) > 1 else float('inf'))
        ]
        performance_ranking = sorted(individuals, key=lambda x: x[1])
        
        # Stage 1: 生成短期反思
        reflection_prompt = self.build_performance_reflection_prompt(
            offspring, parents, performance_ranking
        )
        messages = [
            {"role": "system", "content": self.system_reflector_prompt},
            {"role": "user", "content": reflection_prompt}
        ]
        
        try:
            short_term_reflection = multi_chat_completion([messages], 1, self.cfg.model, self.cfg.temperature)[0]
            logging.info(f"[ReflectiveSearch] Generated short-term reflection for {operator_type}")
        except Exception as e:
            logging.error(f"[ReflectiveSearch] Error generating short-term reflection for {operator_type}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            short_term_reflection = f"Failed to generate reflection due to: {str(e)[:100]}"
        
        # Stage 2: 周期性生成核心原则
        core_principle = None
        if self.iteration % self.reflection_frequency == 0 and self.iteration > 0:
            recent_reflections = self.reflection_lib.get_recent_reflections(n=10)
            if recent_reflections:
                reflections_text = "\n\n".join([
                    f"Operator: {r['operator']}, Gain: {r['gain']:.4f}\n{r['content']}"
                    for r in recent_reflections
                ])
                
                principle_prompt = self.principle_synthesis_template.format(
                    short_term_reflections=reflections_text
                )
                messages = [
                    {"role": "system", "content": self.system_reflector_prompt},
                    {"role": "user", "content": principle_prompt}
                ]
                
                try:
                    core_principle = multi_chat_completion([messages], 1, self.cfg.model, self.cfg.temperature)[0]
                    logging.info(f"[ReflectiveSearch] Generated core principle at iteration {self.iteration}")
                except Exception as e:
                    logging.error(f"[ReflectiveSearch] Error generating core principle at iteration {self.iteration}: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
                    core_principle = None
        
        return short_term_reflection, core_principle
    
    def update_gain_weighted_memory(self):
        """自适应损失计算，更新长期反思"""
        if not self.reflection_enabled:
            return
        
        try:
            recent_reflections = self.reflection_lib.get_recent_reflections(n=self.reflection_frequency * 2)
            if not recent_reflections:
                logging.info("[ReflectiveSearch] No recent reflections available for memory update")
                return
        except Exception as e:
            logging.error(f"[ReflectiveSearch] Error getting recent reflections: {e}")
            return
        
        # 根据性能增益加权反思
        if self.performance_weighted:
            # 计算权重
            gains = [r['gain'] for r in recent_reflections]
            gains_array = np.array(gains)
            # 使用softmax归一化
            exp_gains = np.exp(gains_array - np.max(gains_array))
            weights = exp_gains / np.sum(exp_gains)
            
            # 选择权重较高的反思
            threshold = 1.0 / len(weights) * 0.5  # 低于平均权重一半的被过滤
            filtered_reflections = [
                r for r, w in zip(recent_reflections, weights) if w > threshold
            ]
        else:
            filtered_reflections = recent_reflections
        
        # 构建提示词
        reflections_text = "\n\n".join([
            f"[{r['operator'].upper()}] Gain: {r['gain']:.4f}\n{r['content']}"
            for r in filtered_reflections
        ])
        
        prompt = self.loss_calculate_template.format(
            prior_reflection=self.reflection_lib.long_term_reflection or "No prior reflection.",
            new_reflections=reflections_text
        )
        
        messages = [
            {"role": "system", "content": self.system_reflector_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            updated_reflection = multi_chat_completion([messages], 1, self.cfg.model, self.cfg.temperature)[0]
            
            # ✨ 严格的长度控制和验证
            lines = updated_reflection.strip().split('\n')
            
            # 只保留以"Principle"开头的行
            principle_lines = [line.strip() for line in lines if line.strip().startswith('Principle')]
            
            # 强制限制为3条原则（与prompt一致）
            if len(principle_lines) > 3:
                principle_lines = principle_lines[:3]
                logging.warning(f"[ReflectiveSearch] Truncated long-term reflection from {len(lines)} to 3 principles")
            elif len(principle_lines) < 3:
                logging.warning(f"[ReflectiveSearch] Only got {len(principle_lines)} principles, expected 3")
            
            # 对每条原则进行字数限制（20词）
            truncated_principles = []
            for i, principle in enumerate(principle_lines):
                words = principle.split()
                if len(words) > 25:  # 允许稍微超过20词的限制
                    truncated = ' '.join(words[:25]) + '...'
                    logging.warning(f"[ReflectiveSearch] Principle {i+1} truncated from {len(words)} to 25 words")
                    truncated_principles.append(truncated)
                else:
                    truncated_principles.append(principle)
            
            updated_reflection = '\n'.join(truncated_principles)
            
            # 最终的字符长度检查
            char_count = len(updated_reflection)
            word_count = len(updated_reflection.split())
            
            if char_count > 500:  # ~500字符硬上限（更严格）
                logging.warning(f"[ReflectiveSearch] Reflection too long ({char_count} chars), applying hard truncation...")
                updated_reflection = updated_reflection[:500]
                # 确保不在单词中间截断
                last_space = updated_reflection.rfind(' ')
                if last_space > 400:  # 保留大部分内容
                    updated_reflection = updated_reflection[:last_space] + '...'
            
            self.reflection_lib.long_term_reflection = updated_reflection
            logging.info(f"[ReflectiveSearch] Updated long-term reflection: {len(truncated_principles)} principles, {word_count} words, {char_count} chars")
            
            # 保存到输出目录（使用相对路径）
            reflection_file = os.path.join(self.output_dir, f"long_term_reflection_iter{self.iteration}.txt")
            with open(reflection_file, 'w', encoding='utf-8') as f:
                f.write(updated_reflection)
                f.write(f"\n\n# Metadata\n")
                f.write(f"# Iteration: {self.iteration}\n")
                f.write(f"# Principles: {len(truncated_principles)}\n")
                f.write(f"# Words: {word_count}\n")
                f.write(f"# Chars: {char_count}\n")
            
        except Exception as e:
            logging.error(f"[ReflectiveSearch] Error updating long-term reflection: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    def calculate_performance_gain(self, offspring: Dict, parents: List[Dict]) -> float:
        """计算性能增益"""
        offspring_obj = offspring.get('objective', float('inf'))
        if not parents:
            return 0.0
        
        parent_objs = [p.get('objective', float('inf')) for p in parents]
        best_parent_obj = min(parent_objs)
        
        # 性能增益 = (父代最优 - 子代) / 父代最优
        # 如果子代更好（目标值更小），增益为正
        if best_parent_obj == 0 or best_parent_obj == float('inf'):
            return 0.0
        
        gain = (best_parent_obj - offspring_obj) / abs(best_parent_obj)
        return gain
    
    def adaptive_operator_selection_weights(self) -> Dict[str, float]:
        """基于历史性能自适应调整算子权重"""
        if not self.adaptive_operator_selection:
            return self.operator_weights
        
        operator_scores = self.reflection_lib.calculate_operator_performance()
        
        # 如果没有足够的历史数据，使用默认权重
        if all(score == 0.0 for score in operator_scores.values()):
            return self.operator_weights
        
        # 归一化为概率分布
        min_score = min(operator_scores.values())
        # 将负增益转为正值（加上最小值的绝对值）
        if min_score < 0:
            operator_scores = {k: v - min_score + 0.1 for k, v in operator_scores.items()}
        else:
            operator_scores = {k: v + 0.1 for k, v in operator_scores.items()}
        
        total = sum(operator_scores.values())
        operator_probs = {k: v / total for k, v in operator_scores.items()}
        
        logging.info(f"Adaptive operator weights: {operator_probs}")
        return operator_probs
    
    def _reflection_callback(self, offspring, parents, operator_type, generation):
        """Record performance evidence after each candidate is evaluated.
        
        Args:
            offspring: 生成的子代个体
            parents: 父代个体列表
            operator_type: 使用的算子类型
            generation: 当前代数
        """
        if not self.reflection_enabled:
            return
        
        self.iteration = generation
        
        # Generate a reflection from the measured parent/offspring gain.
        short_term_reflection, core_principle = self.generate_performance_reflection(
            offspring, parents, operator_type
        )
        
        # 存储短期反思
        if short_term_reflection:
            gain = self._calculate_gain(offspring, parents)
            self.reflection_lib.add_reflection(
                reflection=short_term_reflection,
                operator_type=operator_type,
                performance_gain=gain
            )
            logging.info(f"[ReflectiveSearch] [{operator_type}] Reflection stored, gain: {gain:+.2f}")
        
        # 存储核心原则到算子特定指导
        if core_principle:
            self.reflection_lib.update_operator_guidance(operator_type, core_principle)
            logging.info(f"[ReflectiveSearch] Updated core principle for {operator_type} at generation {generation}")
        
        # 周期性执行自适应更新
        if generation % self.reflection_frequency == 0 and generation > 0:
            self.update_gain_weighted_memory()
            
            # 自适应算子权重（未来功能）
            if self.adaptive_operator_selection:
                new_weights = self.adaptive_operator_selection_weights()
                logging.info(f"[ReflectiveSearch] Adaptive operator weights: {new_weights}")
    
    def _calculate_gain(self, offspring, parents):
        """
        计算offspring相对于parents的性能增益
        
        Args:
            offspring: 子代个体
            parents: 父代个体列表
            
        Returns:
            float: 性能增益值（正值表示改进）
        """
        offspring_obj = offspring.get('objective', float('inf'))
        
        if not parents or parents is None:
            return 0.0
        
        parent_objs = [p.get('objective', float('inf')) for p in parents if p is not None]
        if not parent_objs:
            return 0.0
        
        best_parent_obj = min(parent_objs)
        
        # 对于最小化问题，增益 = parent_obj - offspring_obj
        gain = best_parent_obj - offspring_obj
        return gain
    
    def _guidance_provider(self, operator_type):
        """
        Provide reflection-derived guidance for the next candidate proposal.
        
        Args:
            operator_type: 算子类型 (e1, e2, m1, m2)
            
        Returns:
            str: 优化指导文本
        """
        if not self.reflection_enabled:
            return ""
        
        try:
            # 获取算子特定的指导
            guidance = self.reflection_lib.get_guidance_for_operator(operator_type)
            return guidance
        except Exception as e:
            logging.error(f"[ReflectiveSearch] Error getting guidance for operator {operator_type}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return ""
    
    def evolve(self):
        """主进化循环"""
        logging.info("Starting BATTWISE Reflective Metric Search...")
        logging.info(f"Hierarchical reflection enabled: {self.reflection_enabled}")
        logging.info(f"Reflection frequency: {self.reflection_frequency}")
        logging.info(f"Adaptive operator selection: {self.adaptive_operator_selection}")
        
        try:
            # 创建基础EOH实例，传递textgrad回调
            fast_eval = getattr(self.cfg, 'fast_eval', False)
            logging.info(f"Fast eval mode: {fast_eval}")
            
            eoh_method = EOH(
                self.paras, 
                self.problem, 
                prob_rank, 
                pop_greedy, 
                fast_eval=fast_eval,
                # Keyword names belong to the attributed upstream EoH adapter;
                # BATTWISE supplies its reflection hook through that interface.
                textgrad_callback=self._reflection_callback if self.reflection_enabled else None,
                textgrad_guidance_provider=self._guidance_provider if self.reflection_enabled else None,
            )
            
            # 运行EOH并收集结果
            results = eoh_method.run()
        except Exception as e:
            logging.error(f"[BATTWISE] Critical error during reflective search: {e}")
            import traceback
            logging.error(traceback.format_exc())
            # 尝试返回部分结果
            logging.info("[BATTWISE] Attempting to return partial search results...")
            return None, None, float('inf')
        
        # Reflection integration: guidance injection, post-evaluation evidence
        # capture, periodic memory consolidation, and adaptive operator weights.
        
        # 记录最终统计
        try:
            if self.reflection_enabled:
                logging.info("=" * 80)
                logging.info("[ReflectiveSearch] Statistics:")
                logging.info(f"  Total reflections: {len(self.reflection_lib.short_term_reflections)}")
                long_term_len = len(self.reflection_lib.long_term_reflection) if self.reflection_lib.long_term_reflection else 0
                logging.info(f"  Long-term reflection length: {long_term_len} chars")
                logging.info("=" * 80)
        except Exception as e:
            logging.warning(f"[ReflectiveSearch] Error logging statistics: {e}")
        
        logging.info("BATTWISE Reflective Metric Search completed")
        try:
            logging.info(f"Best objective: {results[2] if len(results) > 2 else 'N/A'}")
        except:
            logging.info("Best objective: Unable to retrieve")
        
        return results


def main_test():
    """测试函数"""
    from omegaconf import OmegaConf
    
    # 创建测试配置
    cfg = OmegaConf.create({
        'algorithm': 'reflective_metric_search',
        'model': 'qwen-turbo-latest',
        'temperature': 1.0,
        'max_fe': 10,
        'pop_size': 5,
        'timeout': 200,
        'reflection_enabled': True,
        'reflection_frequency': 3,
        'adaptive_operator_selection': True,
        'performance_weighted': True,
        'max_short_term_reflections': 15,
        'operator_weights': {
            'i1': 0.3,
            'e1': 0.3,
            'e2': 0.0,
            'm1': 0.2,
            'm2': 0.2
        },
        'problem': {
            'problem_name': 'hfsp_constructive',
            'description': 'Hybrid Flow Shop Scheduling Problem',
            'problem_size': 'small',
            'func_name': 'get_combined_expression',
            'obj_type': 'min',
            'problem_type': 'constructive'
        }
    })
    
    root_dir = os.getcwd()
    case_num = [0]
    
    search = ReflectiveMetricSearch(cfg, root_dir, case_num)
    
    # 运行进化
    results = search.evolve()
    
    print("Evolution completed!")
    print(f"Results: {results}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main_test()
