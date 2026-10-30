import re
import time
from .interface_LLM import InterfaceAPI as InterfaceLLM
import re

input = lambda: ...

class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode,prompts, **kwargs):
        # -------------------- RZ: use local LLM --------------------
        assert 'use_local_llm' in kwargs
        assert 'url' in kwargs
        self._use_local_llm = kwargs.get('use_local_llm')
        self._url = kwargs.get('url')
        # -----------------------------------------------------------
        
        # Textgrad guidance provider
        self.guidance_provider = kwargs.get('guidance_provider', None)
        
        # LLM并发控制
        self.llm_max_workers = kwargs.get('llm_max_workers', None)

        # set prompt interface
        #getprompts = GetPrompts()
        self.prompt_task         = prompts.get_task()
        self.prompt_func_name    = prompts.get_func_name()
        self.prompt_func_inputs  = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf    = prompts.get_inout_inf()
        self.prompt_other_inf    = prompts.get_other_inf()
        self.func_signature      = prompts.func_signature
        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode # close prompt checking

        # -------------------- RZ: use local LLM --------------------
        self.interface_llm = InterfaceLLM(
            self.api_endpoint, self.api_key, self.model_LLM, self.debug_mode,
            max_workers=self.llm_max_workers  # 传递LLM并发控制参数
        )

    def get_prompt_i1(self):
        prompt_content = (
            f"{self.prompt_task}\n"
            "First, describe your new algorithm and main steps in one sentence. "
            "The description must be inside a brace. "
            "Next, implement it in Python, providing the complete function. "
            f"The function must have the following signature: `{self.func_signature}`.\n"
            f"{self.prompt_inout_inf}\n"
            f"{self.prompt_other_inf}\n"
            "Your response should contain only the description in braces and the code inside a ```python ... ``` block."
        )
        return prompt_content

        
    def get_prompt_e1(self,indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"

        # 🔄 Textgrad: 获取优化指导
        textgrad_guidance = self._get_textgrad_guidance('e1')
        
        prompt_content = (
            f"{self.prompt_task}\n"
            f"{textgrad_guidance}"  # ✨ 注入textgrad指导
            f"I have {len(indivs)} existing algorithms with their codes as follows: \n"
            f"{prompt_indiv}"
            "Please help me create a new algorithm that has a totally different form from the given ones.\n"
            "First, describe your new algorithm and main steps in one sentence. "
            "The description must be inside a brace. Next, implement it in Python, providing the complete function. "
            f"The function must have the following signature: `{self.func_signature}`.\n"
            f"{self.prompt_inout_inf}\n"
            f"{self.prompt_other_inf}\n"
            "Your response should contain only the description in braces and the code inside a ```python ... ``` block."
        )
        return prompt_content
    
    def _get_textgrad_guidance(self, operator_type):
        """
        获取Textgrad优化指导
        
        Args:
            operator_type: 算子类型 (e1, e2, m1, m2)
            
        Returns:
            str: 优化指导文本（如果有的话）
        """
        if self.guidance_provider is None:
            return ""
        
        try:
            guidance = self.guidance_provider(operator_type)
            if guidance:
                return f"\n📚 **Optimization Guidance from Previous Iterations:**\n{guidance}\n\n"
            return ""
        except Exception as e:
            # 静默失败，不影响正常流程
            return ""
    
    def get_prompt_e2(self,indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv=prompt_indiv+"No."+str(i+1) +" algorithm and the corresponding code are: \n" + indivs[i]['algorithm']+"\n" +indivs[i]['code']+"\n"

        # 🔄 Textgrad: 获取优化指导
        textgrad_guidance = self._get_textgrad_guidance('e2')
        
        prompt_content = (
            f"{self.prompt_task}\n"
            f"{textgrad_guidance}"
            f"I have {len(indivs)} existing algorithms with their codes as follows: \n"
            f"{prompt_indiv}"
            "Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.\n"
            "Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm in one sentence. "
            "The description must be inside a brace. Thirdly, implement it in Python, providing the complete function. "
            f"The function must have the following signature: `{self.func_signature}`.\n"
            f"{self.prompt_inout_inf}\n"
            f"{self.prompt_other_inf}\n"
            "Your response should contain only the description in braces and the code inside a ```python ... ``` block."
        )
        return prompt_content
    
    def get_prompt_m1(self,indiv1):
        # 🔄 Textgrad: 获取优化指导
        textgrad_guidance = self._get_textgrad_guidance('m1')
        
        prompt_content = (
            f"{self.prompt_task}\n"
            f"{textgrad_guidance}"
            "I have one algorithm with its code as follows. "
            f"Algorithm description: {indiv1['algorithm']}\n"
            f"Code:\n{indiv1['code']}\n"
            "Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.\n"
            "First, describe your new algorithm and main steps in one sentence. "
            "The description must be inside a brace. Next, implement it in Python, providing the complete function. "
            f"The function must have the following signature: `{self.func_signature}`.\n"
            f"{self.prompt_inout_inf}\n"
            f"{self.prompt_other_inf}\n"
            "Your response should contain only the description in braces and the code inside a ```python ... ``` block."
        )
        return prompt_content
    
    def get_prompt_m2(self,indiv1):
        # 🔄 Textgrad: 获取优化指导
        textgrad_guidance = self._get_textgrad_guidance('m2')
        
        prompt_content = (
            f"{self.prompt_task}\n"
            f"{textgrad_guidance}"
            "I have one algorithm with its code as follows. "
            f"Algorithm description: {indiv1['algorithm']}\n"
            f"Code:\n{indiv1['code']}\n"
            "Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided.\n"
            "First, describe your new algorithm and main steps in one sentence. "
            "The description must be inside a brace. Next, implement it in Python, providing the complete function. "
            f"The function must have the following signature: `{self.func_signature}`.\n"
            f"{self.prompt_inout_inf}\n"
            f"{self.prompt_other_inf}\n"
            "Your response should contain only the description in braces and the code inside a ```python ... ``` block."
        )
        return prompt_content


    def _get_alg(self,prompt_content):

        response = self.interface_llm.get_response(prompt_content)

        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        
        pattern_code = r'```python(.*?)```'
        code_match = re.search(pattern_code, response, re.DOTALL)
        code = code_match.group(1).strip() if code_match is not None else None

        n_retry = 1
        while code is None:
            if self.debug_mode:
                print("Error: code not identified, wait 1 seconds and retrying ... ")

            if n_retry > 3:
                break
            n_retry +=1
            time.sleep(1)
            
            response = self.interface_llm.get_response(prompt_content)

            algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
            code_match = re.search(pattern_code, response, re.DOTALL)
            code = code_match.group(1).strip() if code_match is not None else None
                
        if code is None:
            return [None, None]

        if not algorithm:
            algorithm_desc = "Algorithm description not found"
        else:
            algorithm_desc = algorithm[0]


        if "np" in code and "import numpy as np" not in code:
            code = "import numpy as np\n" + code

        return [code, algorithm_desc]


    def _get_alg_batch(self, prompt_contents: list[str]):
        responses = self.interface_llm.get_response_batch(prompt_contents)
        results = []
        for response in responses:
            algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)

            pattern_code = r'```python(.*?)```'
            code_match = re.search(pattern_code, response, re.DOTALL)
            code = code_match.group(1).strip() if code_match is not None else None

            if len(algorithm) == 0 or code is None:
                results.append([None, None])
                continue

            algorithm = algorithm[0]

            if "np" in code and "import numpy as np" not in code:
                code = "import numpy as np\n" + code
            
            results.append([code, algorithm])
        return results


    def i1(self):

        prompt_content = self.get_prompt_i1()

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def e1(self,parents):
      
        prompt_content = self.get_prompt_e1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def e2(self,parents):
      
        prompt_content = self.get_prompt_e2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def m1(self,parents):
      
        prompt_content = self.get_prompt_m1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
    
    def m2(self,parents):
      
        prompt_content = self.get_prompt_m2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content )
            print(">>> Press 'Enter' to continue")
            input()
      
        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]