
from datetime import datetime

from src.client import get_clickhouse_client
from src.client.clickhouse_queries import CLICKHOUSE_SESSION_MEMORY



def insert(table, data, columns):
    chc = get_clickhouse_client()
    return chc.insert(table, data, columns)


def write_memory(session_id, user_id, user_question, final_answer, trace_id):
    """只在 CotAgent 完成时调一次"""
    # user 轮
    insert("session_memory", [[
            session_id,
            user_id,
            "user",
            user_question,
            trace_id,
            datetime.now(),
    ]], columns=[
        'session_id',
        'user_id',
        'role',
        'content',
        'trace_id',
        'created_at',
    ])


def get_session_history(session_id: str, n_rounds: int = 5) -> list[dict]:
    """拉最近 n_rounds 轮对话 =2n 条"""
    chc = get_clickhouse_client()
    rows = chc.query(CLICKHOUSE_SESSION_MEMORY,
                     parameters={'session_id': session_id, 'limit': n_rounds * 2})
    return list(rows.named_results())


