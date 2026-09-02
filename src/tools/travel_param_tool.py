import json
import re
from langchain_core.tools import tool

from src.chat.qwen_llm import QwenLlm
from src.prompt.cot_prompt import CotPrompt
from src.service.config import read_environment_config


def _parse_llm_json(content: str) -> dict:
    """健壮地解析 LLM 输出的 JSON"""
    # 1. 尝试直接解析
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 2. 去掉 markdown 代码块（```json ... ```）
    # 匹配第一个 {...} 或 [...] 块
    match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # 3. 实在解析不了，返回空 dict
    print("llm的json无法解析为对应参数")
    return {}


@tool
def get_travel_param(input:str) -> dict:
    """
    当用户问题和旅游攻略有关适合，分析用户意图是否和{地点、时间点、天气、交通、美食、路线有关}，如果是，请按格式返回。
    如：params = {"places":["大阪","京都"], "dates":["2026-08-01","2026-08-02"], "weather":None, "traffic":"JR线", "foods":["汤咖哩","和牛"], "line":""}
    """
    params = """{"places":[], "dates":[], "weather":"", "traffic":"", "foods":[], "line":""}"""
    cp = CotPrompt()
    cp.set_customized_format(params)
    base_url = read_environment_config("OLLAMA_BASE_URL")
    api_key = read_environment_config("OLLAMA_API_KEY")
    model_name = read_environment_config("OLLAMA_MODEL_NAME")
    llm = QwenLlm(api_key, base_url, model_name)
    response = llm.invokeLlm(input=input, base_prompt=cp)
    content = response.content
    # 解析为 dict
    params = _parse_llm_json(content)
    print(f"[DEBUG] raw content: {content}")
    print(f"[DEBUG] parsed: {params}")
    return params