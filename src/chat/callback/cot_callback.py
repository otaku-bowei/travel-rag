

import time
import uuid as uuid_lib
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from overrides import overrides

from src.client import ClickHouseClient


class ClickhouseRecordCoTCallback(BaseCallbackHandler):
    """
    CoT 步骤记录 callback
    - 每次 LLM 调用 = 一个 CoT step
    - thought = LLM 文本输出
    - action = LLM 决定的 tool 调用（如 "rag_search"），没有则为空
    - observation = 本 callback 不写入，留空后续 JOIN tool_calls 关联--TODO
    - step_index 按 trace_id 维度累计，每个新 trace 从 1 开始
    - 失败不能影响主链路
    """

    def __init__(self, chc: ClickHouseClient, trace_id: str = None):
        # ⭐ 不再依赖 current_trace_id，实例自己的 trace_id
        self._trace_id = trace_id or str(uuid_lib.uuid4())
        # run_id → perf_counter 起始时间
        self._start_times: dict = {}
        # trace_id → 当前已记录到的 step_index（同一 trace 累计递增）
        self._step_index: dict = {}
        # run_id → LLM 输出的 tool_calls（用于提取 action）
        self._pending_actions: dict = {}
        self.chc = chc

    @overrides
    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        rid = str(run_id)
        if rid not in self._start_times:
            self._start_times[rid] = time.perf_counter()

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        rid = str(run_id)
        if rid not in self._start_times:
            self._start_times[rid] = time.perf_counter()

    @overrides
    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs):
        rid = str(run_id)
        start = self._start_times.pop(rid, None)
        if start is None:
            return
        duration_ms = int((time.perf_counter() - start) * 1000)

        # ⭐ 用 self._trace_id，不再读 ContextVar
        trace_id = self._trace_id

        thought, action = self._extract_thought_and_action(response)

        # 计算 step_index（按 trace_id 累计）
        idx = self._step_index.get(trace_id, 0) + 1
        self._step_index[trace_id] = idx

        self._insert(
            trace_id=trace_id,
            step_index=idx,
            thought=thought,
            action=action,
            observation=None,  # 不在本 callback 写入，由后续 JOIN tool_calls 获取
            duration_ms=duration_ms,
        )

    @overrides
    def on_llm_error(self, error: BaseException, *, run_id, **kwargs):
        rid = str(run_id)
        start = self._start_times.pop(rid, None)
        if start is None:
            return
        duration_ms = int((time.perf_counter() - start) * 1000)

        # ⭐ 用 self._trace_id
        trace_id = self._trace_id

        idx = self._step_index.get(trace_id, 0) + 1
        self._step_index[trace_id] = idx

        self._insert(
            trace_id=trace_id,
            step_index=idx,
            thought='',
            action='',
            observation=f"ERROR: {str(error)[:300]}",
            duration_ms=duration_ms,
        )


    def _extract_thought_and_action(self, response: LLMResult) -> tuple[str, str]:
        """
        从 LLMResult 提取 thought + action：
        - thought = response.generations[0][0] 的文本内容
        - action = 第一个 tool_call 的名字（如果有）
        """
        thought = ''
        action = ''

        if not response.generations:
            return thought, action

        gen_list = response.generations[0]  # 第一批（通常是单批）
        if not gen_list:
            return thought, action

        gen = gen_list[0]
        # gen 可能是 ChatGeneration（有 message 属性）或 LLMResult.text
        # 优先取 message.content（ChatModel 标准）
        message = getattr(gen, 'message', None)
        if message is not None:
            content = getattr(message, 'content', '')
            thought = content if isinstance(content, str) else str(content)
            # 取 tool_calls
            tool_calls = getattr(message, 'tool_calls', None)
            if tool_calls:
                action = self._format_action(tool_calls)
        else:
            # fallback：直接读 text
            thought = getattr(gen, 'text', '') or ''

        # 截断（防超长）
        thought = thought[:2000] if thought else ''
        action = action[:500] if action else ''
        return thought, action

    def _format_action(self, tool_calls: list) -> str:
        """
        格式化 tool_call 为 "tool_name(arg1=val1, arg2=val2)"
        兼容三种格式：
        - LangChain 1.0+ 标准：{'name': ..., 'args': ..., 'id': ..., 'type': 'tool_call'}
        - OpenAI 原始：{'function': {'name': ..., 'arguments': '...'}}
        - 旧版简化：{'name': ..., 'args': ...}
        """
        if not tool_calls:
            return ''
        first_tc = tool_calls[0]
        name = ''
        args = ''

        if isinstance(first_tc, dict):
            # OpenAI 原始格式：function.name + function.arguments
            func = first_tc.get('function')
            if isinstance(func, dict) and func.get('name'):
                name = func['name']
                raw_args = func.get('arguments', '')
                args = raw_args if isinstance(raw_args, str) else str(raw_args)
            else:
                # LangChain 1.0+ 标准格式或旧版简化格式
                name = first_tc.get('name', '') or ''
                raw_args = first_tc.get('args', '')
                args = str(raw_args) if raw_args else ''

        if not name:
            return ''
        # 简化：args 太长就截断
        if args and len(args) > 200:
            args = args[:200] + '...'
        return f"{name}({args})" if args else name

    def _insert(self, trace_id, step_index, thought, action,
                observation, duration_ms):
        """
        写入 cot_steps 表

        注意：observation 字段在当前实现中不写入（传 None），
        需要后续通过 JOIN tool_calls 表获取实际 tool 执行结果：
        SELECT * FROM cot_steps cs
        LEFT JOIN tool_calls tc
          ON cs.trace_id = tc.trace_id
         AND cs.action LIKE concat('%', tc.tool_name, '%')
        """
        try:
            self.chc.insert(
                table='cot_steps',
                data=[[
                    trace_id,
                    int(step_index),
                    thought,
                    action,
                    observation,
                    duration_ms,
                ]],
                columns=[
                    'trace_id',
                    'step_index',
                    'thought',
                    'action',
                    'observation',
                    'duration_ms',
                ]
            )
        except Exception as e:
            import traceback
            print(f"[ClickhouseRecordCoTCallback] 写入失败: {e}")
            traceback.print_exc()