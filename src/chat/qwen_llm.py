from src.chat.base import Llm

"""
本地ollama安装的deepseek的llm模型
"""
class DeepseekLlm(Llm):

    def __init__(self, model_name = "qwen", base_url : str, ):
        super().__init__()
        self.base_url = base_url
        