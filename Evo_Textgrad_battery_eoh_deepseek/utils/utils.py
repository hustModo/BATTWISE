import subprocess
import os
import json
import logging
import concurrent.futures
import time
import re
import inspect

# Full EOH runs in a child process.  The launcher passes the application's
# resolved settings through these variables, while standalone users can set
# them directly.  Credentials must never be embedded in source.
def _environment_api_keys():
    raw_many = os.environ.get("BATRECYCLER_EOH_API_KEYS", "")
    candidates = re.split(r"[,;\r\n]+", raw_many) if raw_many else []
    candidates.extend(
        [
            os.environ.get("BATRECYCLER_EOH_API_KEY", ""),
            os.environ.get("OPENAI_API_KEY", ""),
        ]
    )
    return list(dict.fromkeys(key.strip() for key in candidates if key.strip()))


_API_KEYS = _environment_api_keys()
_BASE_URL = (
    os.environ.get("BATRECYCLER_EOH_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL")
    or ""
)
API_CONFIGS = {
    provider: {"base_url": _BASE_URL, "api_keys": list(_API_KEYS)}
    for provider in ("gpt", "deepseek", "qwen")
}

def get_api_config(model_name):
    """根据模型名称获取API配置"""
    if model_name.startswith("gpt"):
        provider = "gpt"
    elif model_name.startswith("deepseek"):
        provider = "deepseek"
    else:
        provider = "qwen"
    config = API_CONFIGS[provider]
    if not config["api_keys"]:
        raise RuntimeError(
            "No Full EOH LLM credential configured. Set "
            "BATRECYCLER_EOH_API_KEY(S) or launch EOH through the application."
        )
    return config

def init_client(cfg):
    """初始化默认API客户端"""
    global client
    api_config = get_api_config(cfg.model)
    
    from openai import OpenAI
    client = OpenAI(
        api_key=api_config["api_keys"][0],  # 使用第一个密钥
        base_url=api_config["base_url"] or None
    )
    
    # 特殊处理GPT模型
    if cfg.model.startswith("gpt"):
        import openai
        openai.api_base = api_config["base_url"]

def init_island_client(cfg, island_id):
    """为每个岛屿初始化独立的API客户端"""
    global client
    api_config = get_api_config(cfg.model)
    
    # 获取岛屿特定的API密钥
    api_key_index = island_id % len(api_config["api_keys"])
    api_key = api_config["api_keys"][api_key_index]
    
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url=api_config["base_url"] or None
    )
    
    # 特殊处理GPT模型
    if cfg.model.startswith("gpt"):
        import openai
        openai.api_base = api_config["base_url"]
    
    logging.info(f"Island {island_id}: Initialized with API key {api_key_index}")


def file_to_string(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found

def block_until_running(stdout_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the evaluation has started before moving on
    while True:
        log = file_to_string(stdout_filepath)
        if  len(log) > 0:
            if log_status and "Traceback" in log:  # 如果存在则会出现"Traceback"
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            else:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successful!")
            break


def extract_description(response: str) -> tuple[str, str]:
    # Regex patterns to extract code description enclosed in GPT response, it starts with ‘<start>’ and ends with ‘<end>’
    pattern_desc = [r'<start>(.*?)```python', r'<start>(.*?)<end>']
    for pattern in pattern_desc:
        desc_string = re.search(pattern, response, re.DOTALL)
        desc_string = desc_string.group(1).strip() if desc_string is not None else None
        if desc_string is not None:
            break
    return desc_string


def multi_chat_completion(messages_list: list[list[dict]], n, model, temperature, island_id: int = None, max_workers: int = None, use_api_n_param: bool = True):
    """
    批量并行调用LLM，支持并发控制和API n参数优化
    
    An example of messages_list:
    
    messages_list = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
        [
            {"role": "system", "content": "You are a knowledgeable guide."},
            {"role": "user", "content": "How are you?"},
        ],
        [
            {"role": "system", "content": "You are a witty comedian."},
            {"role": "user", "content": "Tell me a joke."},
        ]
    ]
    
    Args:
        n: number of responses to generate for each message in messages_list
        island_id: island ID for API client identification
        max_workers: 最大并发worker数，None则根据模型自动设置
                    - GPT: 5 (限制较严)
                    - Qwen/通义千问: 10 (较宽松)
                    - Claude: 8 (中等)
                    - 其他: 10 (默认)
        use_api_n_param: 是否使用API的n参数优化（推荐True）
                        True: 使用混合方式（多线程 + API n参数）
                        False: 纯多线程方式（兼容模式）
    """
    # If messages_list is not a list of list (i.e., only one conversation), convert it to a list of list
    assert isinstance(messages_list, list), "messages_list should be a list."
    if not isinstance(messages_list[0], list):
        messages_list = [messages_list]
    
    # 确定API的n参数最大值（根据模型）
    max_n_per_call = 1
    if use_api_n_param and "gpt" in model.lower():
        max_n_per_call = 10  # GPT支持较大的n
    elif use_api_n_param and ("qwen" in model.lower() or "通义" in model.lower()):
        max_n_per_call = 4   # Qwen最大支持n=4
    elif use_api_n_param and "claude" in model.lower():
        max_n_per_call = 5   # Claude估计支持较大的n
    
    # 如果n > 1且启用API n参数优化，使用混合方式
    if n > 1 and max_n_per_call > 1:
        # 混合方式：减少线程数，增加每次调用的n
        actual_n_per_call = min(n, max_n_per_call)
        # 每个message需要调用的次数
        calls_per_message = (n + actual_n_per_call - 1) // actual_n_per_call
        total_calls = len(messages_list) * calls_per_message
        
        logging.info(f"LLM hybrid mode: {len(messages_list)} messages × {n} responses = {len(messages_list)*n} total")
        logging.info(f"  Using {total_calls} API calls with n={actual_n_per_call} each")
        logging.info(f"  Savings: {len(messages_list)*n - total_calls} fewer API calls vs pure multi-threading")
        
        # 构建调用参数：每个message调用calls_per_message次，每次n=actual_n_per_call
        call_args = []
        for messages in messages_list:
            for _ in range(calls_per_message):
                call_args.append((actual_n_per_call, messages, model, temperature, island_id))
        
    elif "gpt" not in model:
        # 非GPT模型，或不使用n参数优化：转换为多次调用
        messages_list *= n
        n = 1
        call_args = [(n, messages, model, temperature, island_id) for messages in messages_list]
        logging.info(f"LLM multi-threading mode: {len(call_args)} API calls with n=1 each")
    else:
        call_args = [(n, messages, model, temperature, island_id) for messages in messages_list]
        logging.info(f"LLM batch mode: {len(call_args)} API calls")

    # 根据模型自动设置并发数（避免触发API限流）
    if max_workers is None:
        if "gpt" in model.lower():
            max_workers = 5  # GPT限制较严
        elif "qwen" in model.lower() or "通义" in model.lower():
            # Qwen并发能力较强，支持高并发
            max_workers = 20  # 通义千问较宽松，适合pop_size=20
        elif "claude" in model.lower():
            max_workers = 8  # Claude中等
        elif "deepseek" in model.lower():
            max_workers = 12  # DeepSeek中等
        else:
            max_workers = 10  # 默认
    
    logging.info(f"LLM batch completion: {len(call_args)} API calls with max_workers={max_workers}")
    
    # 性能监控
    import time
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        choices = executor.map(lambda p: chat_completion(*p), call_args)
    
    # 转换为列表以触发执行
    choices = list(choices)
    elapsed_time = time.time() - start_time
    
    logging.info(f"LLM batch completion time: {elapsed_time:.2f}s ({len(call_args)/elapsed_time:.2f} calls/s)")

    contents: list[str] = []
    for choice in choices:
        if choice is not None:  # 添加None检查
            for c in choice:
                contents.append(c.message.content)
    
    logging.info(f"LLM batch completion finished: {len(contents)} responses generated in {elapsed_time:.2f}s")
    return contents


def chat_completion(n: int, messages: list[dict], model: str, temperature: float, island_id: int = None) -> list[dict]:
    """
    Generate n responses using OpenAI Chat Completions API
    """
    global client
    
    # 如果client未定义，尝试重新初始化
    if 'client' not in globals() or client is None:
        logging.info("Client not initialized, attempting to reinitialize...")
        from openai import OpenAI
        api_config = get_api_config(model)
        client = OpenAI(
            api_key=api_config["api_keys"][0],
            base_url=api_config["base_url"] or None,
        )

    for attempt in range(1000):
        try:
            if "gpt" in model:
                response_cur = client.chat.completions.create(model=model, messages=messages, temperature=temperature, n=n)
            else:
                assert n == 1
                if "GLM" in model:
                    response_cur = client.chat.completions.create(model=model, messages=messages, temperature=min(temperature, 1.))
                else:
                    response_cur = client.chat.completions.create(model=model, messages=messages, temperature=min(temperature, 1.))
            break
        except Exception as e:
            logging.info(f"Island {island_id}: Attempt {attempt+1} failed with error: {e}")
            time.sleep(1)
    if response_cur is None:
        logging.info(f"Island {island_id}: Code terminated due to too many failed attempts!")
        exit()
            
    return response_cur.choices


def extract_code_from_generator(content):
    """Extract code from the response of the code generator."""
    pattern_code = r'```python(.*?)```'
    code_string = re.search(pattern_code, content, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    if code_string is None:
        # Find the line that starts with "def" and the line that starts with "return", and extract the code in between
        lines = content.split('\n')
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.startswith('def'):
                start = i
            if 'return' in line:
                end = i
                break
        if start is not None and end is not None:
            code_string = '\n'.join(lines[start:end+1])
    
    if code_string is None:
        return None
    # Add import statements if not present
    if "np" in code_string:
        code_string = "import numpy as np\n" + code_string
    if "torch" in code_string:
        code_string = "import torch\n" + code_string
    return code_string


def filter_code(code_string):
    """Remove lines containing signature and import statements."""
    lines = code_string.split('\n')
    filtered_lines = []
    for line in lines:
        if line.startswith('def'):
            continue
        elif line.startswith('import'):
            continue
        elif line.startswith('from'):
            continue
        elif line.startswith('return'):
            filtered_lines.append(line)
            break
        else:
            filtered_lines.append(line)
    code_string = '\n'.join(filtered_lines)
    return code_string


def get_heuristic_name(module, possible_names: list[str]):
    for func_name in possible_names:
        if hasattr(module, func_name):
            if inspect.isfunction(getattr(module, func_name)):
                return func_name

def extract_to_hs(input_string: str):
    code_blocks = input_string.split("```python\n")[1:]

    try:
        parameter_ranges_block = "import numpy as np\n" + code_blocks[1].split("```")[0].strip()
        if any(keyword in parameter_ranges_block for keyword in ['inf', 'np.inf', 'None']):
            return None, None
        exec_globals = {}
        exec(parameter_ranges_block, exec_globals)
        parameter_ranges = exec_globals['parameter_ranges']
    except:
        return None, None

    function_block = code_blocks[0].split("```")[0].strip()

    paren_count = 0
    in_signature = False
    signature_start_index = None
    signature_end_index = None

    # Loop through the function block to find the start and end of the function signature
    for i, char in enumerate(function_block):
        if char == "d" and function_block[i:i + 3] == 'def':
            in_signature = True
            signature_start_index = i
        if in_signature:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            if char == ':' and paren_count == 0:
                signature_end_index = i
                break

    if signature_start_index is not None and signature_end_index is not None:
        function_signature = function_block[signature_start_index:signature_end_index + 1]
        for param in parameter_ranges:
            pattern = rf"(\b{param}\b[^=]*=)[^,)]+"
            replacement = r"\1 {" + param + "}"
            function_signature = re.sub(pattern, replacement, function_signature, flags=re.DOTALL)
        function_block = function_block[:signature_start_index] + function_signature + function_block[
                                                                                       signature_end_index + 1:]

    return parameter_ranges, function_block

def format_messages(cfg, pre_messages):
    messages = [{"role": "system", "content": pre_messages["system"]},
                {"role": "user", "content": pre_messages["user"]}]
    return messages
