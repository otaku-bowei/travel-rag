from typing import Any
from uuid import UUID
import time
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from overrides import overrides

from src.client import ClickHouseClient
from src.client.clickhouse_queries import CLICKHOUSE_TOOL_RECORD
from src.monitor.trace_context import current_trace_id


class ClickhouseRecordToolCallback(BaseCallbackHandler):
    """
    工具调用记录，失败或成功时候记录到ch
    """

    def __init__(self, chc : ClickHouseClient):
        self._start_times = {}
        self.chc = chc

    @overrides
    def on_tool_start(self, serialized, input_str, run_id=None, **kwargs):
        tool_name = serialized.get('name', 'unknown')
        self._start_times[str(run_id)] = (time.time(), tool_name)

    @overrides
    def on_tool_end(self, output, run_id=None, input_str=None, **kwargs):
        run_id_str = str(run_id)
        start_info = self._start_times.pop(run_id_str, (time.time(), 'unknown'))
        duration_ms = int((time.time() - start_info[0]) * 1000)
        trace_id = current_trace_id.get()
        if not trace_id:
            return  # 没有 trace_id，跳过

        self._insert(
            trace_id=trace_id,
            tool_call_id=run_id_str,
            tool_name=start_info[1],
            tool_input=input_str if input_str else '',  # 注意：end 时拿不到 input
            tool_output=str(output)[:1000],
            duration_ms=duration_ms,
            success=1,
            error_msg=''
        )
        


    @overrides
    def on_tool_error(self, error, run_id=None, **kwargs):
        run_id_str = str(run_id)
        start_info = self._start_times.pop(run_id_str, (time.time(), 'unknown'))
        duration_ms = int((time.time() - start_info[0]) * 1000)

        trace_id = current_trace_id.get()
        if not trace_id:
            return

        self._insert(
            trace_id=trace_id,
            tool_call_id=run_id_str,
            tool_name=start_info[1],
            tool_input='',
            tool_output='',
            duration_ms=duration_ms,
            success=0,
            error_msg=str(error)[:500]
        )

    def _insert(self, trace_id, tool_call_id, tool_name, tool_input, tool_output, duration_ms, success, error_msg):
        """
        写入一条工具调用记录
        """
        try:
            self.chc.insert(
                table='tool_calls',
                data=[[trace_id,
                       tool_call_id,
                       tool_name,
                       tool_input[:500] if tool_input else '',
                       tool_output[:1000] if tool_output else '',
                       duration_ms,
                       success,
                       error_msg
                       ]],
                columns=[
                    'trace_id',
                    'tool_call_id',
                    'tool_name',
                    'tool_input',
                    'tool_output',
                    'duration_ms',
                    'success',
                    'error_msg'
                ]
            )
        except Exception as e:
            print(f"[ClickhouseRecordToolCallback] 写入失败: {e}")