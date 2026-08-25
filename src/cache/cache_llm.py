from src.cache.redis_cache import RedisCache
from src.chat.minimax_llm import MiniMaxLlm
from src.chat.qwen_llm import QwenLlm
from src.service.config import read_environment_config


_llm_cache = RedisCache(prefix="LLM")

_cot_cache_key = "CoT"
_travel_cache_key = "travel"

def init_cot_llm():
    """
    初始化CoT模型
    """
    base_url = read_environment_config("OLLAMA_BASE_URL")
    api_key = read_environment_config("OLLAMA_API_KEY")
    model_name = read_environment_config("OLLAMA_MODEL_NAME")
    llm = QwenLlm(api_key, base_url, model_name)
    _llm_cache.save_config(cache_key=_cot_cache_key, config=llm.to_dict)

def init_travel_llm():
    """
        初始化旅游模型
    """
    api_key = read_environment_config("MINIMAX_API_KEY")
    model_name = read_environment_config("MINIMAX_MODEL")
    base_url = read_environment_config("MINIMAX_BASE_URL")
    llm = MiniMaxLlm(api_key, base_url, model_name, )
    _llm_cache.save_config(cache_key=_travel_cache_key, config=llm.to_dict)

def get_cot_llm() -> QwenLlm:
    return QwenLlm.from_dict(_llm_cache.get_config(_cot_cache_key))

def get_travel_llm() -> MiniMaxLlm:
    return MiniMaxLlm.from_dict(_llm_cache.get_config(_travel_cache_key))