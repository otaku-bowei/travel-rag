

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import LLMResult
from overrides import overrides

from src.client import ClickHouseClient
from src.monitor.trace_context import current_trace_id


class ClickhouseRecordReactCallback(BaseCallbackHandler):
    """
    ReAct 链路记录 callback
    - 必须依赖 current_trace_id（由 TraceChainCallBack 设置）
    - 失败不能影响主链路（try/except 包住 _insert）
    """

    def __init__(self, chc: ClickHouseClient):
        self._start_times: dict = {}
        self._user_questions: dict = {}
        # run_id → {prompt_tokens, completion_tokens, total_tokens}
        self._token_usage: dict = {}
        # run_id → model_name
        self._llm_models: dict = {}
        self.chc = chc
        self._model_name = "unknow"


    @overrides
    def on_chain_start(self, serialized, inputs, *, run_id, **kwargs):
        rid = str(run_id)
        self._start_times[rid] = time.perf_counter()
        self._token_usage[rid] = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
        }
        self._llm_models[rid] = ''

        # 提取 user_question（最后一个 HumanMessage 的 content）
        user_question = self._extract_user_question(inputs)
        self._user_questions[rid] = user_question

    @overrides
    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        if parent_run_id is None:
            return
        chain_rid = str(parent_run_id)
        if chain_rid not in self._start_times:
            return
        model_name = ''
        if isinstance(serialized, dict):
            model_name = serialized.get('name', '') or ''
        if not model_name:
            metadata = kwargs.get('metadata') or {}
            model_name = metadata.get('ls_model_name', '') or ''
        if model_name and not self._llm_models.get(chain_rid):
            self._llm_models[chain_rid] = model_name

    @overrides
    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        if parent_run_id is None:
            return
        chain_rid = str(parent_run_id)
        if chain_rid not in self._start_times:
            return
        if not self._llm_models.get(chain_rid) and isinstance(serialized, dict):
            model_name = serialized.get('name', '') or ''
            self._model_name = model_name
            if model_name:
                self._llm_models[chain_rid] = model_name

    @overrides
    def on_llm_end(self, response: LLMResult, *, run_id, parent_run_id=None, **kwargs):
        if parent_run_id is None:
            return
        chain_rid = str(parent_run_id)
        if chain_rid not in self._start_times:
            return

        llm_output = getattr(response, 'llm_output', None) or {}
        token_usage = llm_output.get('token_usage') or {}
        if token_usage:
            cur = self._token_usage.get(
                chain_rid,
                {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            )
            cur['prompt_tokens'] += (token_usage.get('prompt_tokens', 0) or 0)
            cur['completion_tokens'] += (token_usage.get('completion_tokens', 0) or 0)
            cur['total_tokens'] += (token_usage.get('total_tokens', 0) or 0)
            self._token_usage[chain_rid] = cur

        if isinstance(llm_output, dict):
            model_name = llm_output.get('model_name', '') or ''
            self._model_name = model_name
            if model_name and not self._llm_models.get(chain_rid):
                self._llm_models[chain_rid] = model_name

    # ============================================================
    # 链路结束（写入 react_chains 主记录）
    # ============================================================
    @overrides
    def on_chain_end(self, outputs, *, run_id, **kwargs):
        rid = str(run_id)
        start = self._start_times.pop(rid, None)
        if start is None:
            return
        duration_ms = int((time.perf_counter() - start) * 1000)

        user_question = self._user_questions.pop(rid, '')
        token_usage = self._token_usage.pop(rid, {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})
        llm_model = self._model_name
        trace_id = current_trace_id.get()
        if not trace_id:
            return

        # 提取 final_answer 和统计 tool / llm step
        final_answer, tool_call_count, cot_step_count, tool_names = self._analyze_outputs(outputs)

        self._insert(
            trace_id=trace_id,
            user_question=user_question,
            final_answer=final_answer,
            tool_call_count=tool_call_count,
            tool_names=tool_names,
            cot_step_count=cot_step_count,
            llm_model=llm_model,
            prompt_tokens=token_usage['prompt_tokens'],
            completion_tokens=token_usage['completion_tokens'],
            total_tokens=token_usage['total_tokens'],
            duration_ms=duration_ms,
            success=1,
            error_msg=''
        )

    @overrides
    def on_chain_error(self, error: BaseException, *, run_id, **kwargs):
        rid = str(run_id)
        start = self._start_times.pop(rid, None)
        if start is None:
            return
        duration_ms = int((time.perf_counter() - start) * 1000)

        user_question = self._user_questions.pop(rid, '')
        # 出错时 token 统计可能不完整
        token_usage = self._token_usage.pop(rid, {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0})
        # llm_model = self._llm_models.pop(rid, '')
        print(kwargs)
        metadata = kwargs.get('metadata') or {}
        llm_model = metadata.get('ls_model_name', '')
        trace_id = current_trace_id.get()
        if not trace_id:
            return

        self._insert(
            trace_id=trace_id,
            user_question=user_question,
            final_answer='',
            tool_call_count=0,
            tool_names=[],
            cot_step_count=0,
            llm_model=llm_model,
            prompt_tokens=token_usage['prompt_tokens'],
            completion_tokens=token_usage['completion_tokens'],
            total_tokens=token_usage['total_tokens'],
            duration_ms=duration_ms,
            success=0,
            error_msg=str(error)[:500]
        )

    # ============================================================
    # 私有工具方法
    # ============================================================
    def _extract_user_question(self, inputs: Any) -> str:
        """
        从 inputs 中提取用户问题（最后一个 HumanMessage 的 content）
        """
        user_question = ''
        if isinstance(inputs, dict):
            # LangGraph create_agent 的 inputs 格式：{"messages": [SystemMessage, ..., HumanMessage]}
            messages = inputs.get('messages') or inputs.get('input')
            if isinstance(messages, list):
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        user_question = msg.content if isinstance(msg.content, str) else str(msg.content)
                        break
            elif isinstance(messages, str):
                user_question = messages
        return user_question[:1000] if user_question else ''

    def _analyze_outputs(self, outputs: Any) -> tuple:
        """
        分析 outputs：
        - final_answer：最后一个 AIMessage 的 content
        - tool_call_count：所有 AIMessage 上 tool_calls 总数
        - cot_step_count：AIMessage 的总数（= LLM 调用次数）
        - tool_names：所有 tool 调用的名称列表
        """
        final_answer = ''
        tool_call_count = 0
        cot_step_count = 0
        tool_names = []

        if not isinstance(outputs, dict):
            return final_answer, tool_call_count, cot_step_count, tool_names

        messages = outputs.get('messages') or outputs.get('output')
        if not isinstance(messages, list):
            return final_answer, tool_call_count, cot_step_count, tool_names

        # 最后一个 AIMessage 作为 final_answer
        # TODO--这里考虑做缓存，包括向量库的一些处理
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = msg.content
                final_answer = content if isinstance(content, str) else str(content)
                break

        # 统计
        for msg in messages:
            if isinstance(msg, AIMessage):
                cot_step_count += 1
                # 提取 tool_calls
                tool_calls = getattr(msg, 'tool_calls', None)
                if not tool_calls:
                    tool_calls = (msg.additional_kwargs or {}).get('tool_calls') or []
                for tc in tool_calls:
                    tool_call_count += 1
                    name = self._extract_tool_name(tc)
                    if name:
                        tool_names.append(name)

        # 限制大小
        tool_names = tool_names[:20]
        final_answer = final_answer[:5000] if final_answer else ''
        return final_answer, tool_call_count, cot_step_count, tool_names

    def _extract_tool_name(self, tool_call: Any) -> str:
        """
        兼容不同格式的 tool_call
        """
        if isinstance(tool_call, dict):
            # OpenAI 格式：{"function": {"name": "...", "arguments": "..."}}
            func = tool_call.get('function') or {}
            if isinstance(func, dict) and func.get('name'):
                return func['name']
            # 简化的格式：{"name": "...", "args": {...}}
            if tool_call.get('name'):
                return tool_call['name']
        return ''

    def _insert(self, trace_id, user_question, final_answer,
                tool_call_count, tool_names, cot_step_count,
                llm_model, prompt_tokens, completion_tokens, total_tokens,
                duration_ms, success, error_msg):
        """
        记录到clickhouse写入 react_chains 表
        """
        try:
            self.chc.insert(
                table='react_chains',
                data=[[
                    trace_id,
                    user_question,
                    final_answer,
                    int(tool_call_count),
                    list(tool_names) if tool_names else [],
                    int(cot_step_count),
                    llm_model or '',
                    int(prompt_tokens) if prompt_tokens is not None else 0,
                    int(completion_tokens) if completion_tokens is not None else 0,
                    int(total_tokens) if total_tokens is not None else 0,
                    int(duration_ms) if duration_ms is not None else 0,
                    int(success),
                    error_msg or '',
                ]],
                columns=[
                    'trace_id',
                    'user_question',
                    'final_answer',
                    'tool_call_count',
                    'tool_names',
                    'cot_step_count',
                    'llm_model',
                    'prompt_tokens',
                    'completion_tokens',
                    'total_tokens',
                    'duration_ms',
                    'success',
                    'error_msg',
                ]
            )
        except Exception as e:
            print(f"[ClickhouseRecordReactCallback] 写入失败: {e}")