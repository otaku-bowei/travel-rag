"""
FastAPI 服务：把主 agent 暴露成 HTTP API
"""
import asyncio
from typing import Any, Annotated

from fastapi import FastAPI, Query

from src.client.model.chat_model import ChatResponse, ChatRequest
from src.service.agent_service import for_one_answer

app = FastAPI(title="Travel Agent Fast API")


@app.post("/chat/v1", response_model=ChatResponse)
async def chat(req: ChatRequest, q: Annotated[str | None, Query()] = None):
    """
    提供http入口用户提问问题
    """
    resp = await for_one_answer(req)
    return resp


@app.get("/health")
async def health():
    return {"status": "ok"}
