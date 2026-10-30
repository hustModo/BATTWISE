import http.client
import json

from utils.utils import chat_completion, multi_chat_completion


class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode, max_workers=None):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 5
        self.max_workers = max_workers  # LLM并发worker数

    def get_response(self, prompt_content):
        # payload_explanation = json.dumps(
        #     {
        #         "model": self.model_LLM,
        #         "messages": [
        #             # {"role": "system", "content": "You are a helpful assistant."},
        #             {"role": "user", "content": prompt_content}
        #         ],
        #     }
        # )

        # headers = {
        #     "Authorization": "Bearer " + self.api_key,
        #     "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
        #     "Content-Type": "application/json",
        #     "x-api2d-no-cache": 1,
        # }
        
        # response = None
        # n_trial = 1
        # while True:
        #     n_trial += 1
        #     if n_trial > self.n_trial:
        #         return response
        #     try:
        #         conn = http.client.HTTPSConnection(self.api_endpoint)
        #         conn.request("POST", "/v1/chat/completions", payload_explanation, headers)
        #         res = conn.getresponse()
        #         data = res.read()
        #         json_data = json.loads(data)
        #         response = json_data["choices"][0]["message"]["content"]
        #         break
        #     except Exception as e:
        #         if self.debug_mode:
        #             print(e)
        #             print("Error in API. Restarting the process...")
        #         continue
            

        # return response

        response = chat_completion(1, [{"role": "user", "content": prompt_content}], self.model_LLM, temperature=1.)
        ret = response[0].message.content
        return ret

    def get_response_batch(self, prompt_contents: list[str]):
        messages_list = [[{"role": "user", "content": prompt}] for prompt in prompt_contents]
        # 传递max_workers参数以控制并发数
        responses = multi_chat_completion(
            messages_list, 1, self.model_LLM, 
            temperature=1., 
            max_workers=self.max_workers
        )
        return responses