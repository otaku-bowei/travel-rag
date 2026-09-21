import asyncio
import threading

from src.agent.cot_agent import CotAgent as CA
from src.cache.cache_llm import get_cot_llm as gcl
from src.client.model.chat_model import ChatRequest, ChatResponse
from src.prompt.cot_prompt import CotPrompt
from src.service.clickhouse_service import write_memory, get_session_history
from src.tools.common_agent_tool import common_agent_tool
from src.tools.rag_tool import travel_rag_search
from src.tools.travel_agent_tool import travel_agent_tool
from src.tools.travel_param_tool import get_travel_param

_cot_agent = None


def get_master_agent():
    global _cot_agent
    if _cot_agent is None:
        cot_llm = gcl([
            # travel_rag_search,
            # get_travel_param,
            travel_agent_tool,
            common_agent_tool,
        ])
        _cot_agent = CA(cot_llm, tools=[
            # travel_rag_search,
            # get_travel_param,
            travel_agent_tool,
            common_agent_tool,
        ])
    return _cot_agent


async def for_one_answer(req: ChatRequest) -> ChatResponse:
    """
    一次性回答用户的问题
    """
    print(f"用户提出了一个问题:{req.question}")
    if not req.question:
        return ChatResponse(answer="请描述你的问题", session_id=req.session_id, )
    try:
        # 9.17 做上下文记忆，先读取，再异步写入新的
        histories = get_session_history(req.session_id, 5)
        session_histories = []
        for h in histories:
            if h['content'] not in session_histories:
                session_histories.append(h['content'])
        threading.Thread(target=write_memory,
                         kwargs={"session_id":req.session_id, "user_id":"", "user_question":req.question, "trace_id":"", "final_answer":""},
                         daemon=True).start()
        # 提示词工程，提问llm
        # bp = CotPrompt()
        bp = CotPrompt(session_histories)
        responses = await get_master_agent().ainvoke(query=req.question, base_prompt=bp)
        print(responses)
        # 取最后一条 AI 消息作为回答
        answer = responses["messages"][-1].content
        return ChatResponse(answer=answer, session_id=req.session_id)
    except Exception as e:
        print(f"\n[错误] {e}\n")
        return ChatResponse(answer="发生异常", session_id=req.session_id)
