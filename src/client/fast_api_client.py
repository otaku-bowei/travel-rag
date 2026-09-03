"""
FastAPI 服务：把主 agent 暴露成 HTTP API
"""
from fastapi import FastAPI
from pydantic import BaseModel

from src.agent.cot_agent import CotAgent
from src.cache.cache_llm import init_cot_llm, get_cot_llm
from src.prompt.cot_prompt import CotPrompt
from src.tools.rag_tool import rag_search
from src.tools.travel_agent_tool import travel_agent_tool
from src.tools.travel_param_tool import get_travel_param

app = FastAPI(title="Travel Agent Fast API")
cot_llm = get_cot_llm([rag_search, get_travel_param, travel_agent_tool])
master_agent = CotAgent(cot_llm, tools=[rag_search, get_travel_param, travel_agent_tool])


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    session_id: str


@app.post("/chat/v1", response_model=ChatResponse)
async def chat(req: ChatRequest):
    bp = CotPrompt()
    result = master_agent.invoke(
        input=req.question,
        base_prompt=bp,
        session_id=req.session_id,
    )
    return ChatResponse(answer=result.content, session_id=req.session_id)


@app.get("/health")
async def health():
    return {"status": "ok"}
