"""
日期工具：获取今天的日期、近期的日期范围
"""
from datetime import datetime, timedelta
from typing import Optional

from langchain_core.tools import tool


def _format_date(d: datetime) -> str:
    """格式化为 YYYY-MM-DD"""
    return d.strftime("%Y-%m-%d")


def _format_weekday_cn(d: datetime) -> str:
    """中文星期几"""
    mapping = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return mapping[d.weekday()]


def _resolve_anchor_date(anchor: Optional[str]) -> datetime:
    """
    解析锚点日期（用于测试或"以未来某天为基准"场景）
    - None / 空 / "today" → 用今天
    - "YYYY-MM-DD" → 用该日期
    - 其他 → 抛 ValueError
    """
    if not anchor or anchor == "today":
        return datetime.now()
    try:
        return datetime.strptime(anchor, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(
            f"日期格式错误: {anchor}，应为 YYYY-MM-DD 或 'today'"
        ) from e


@tool
def get_date_info(
    days_back: int = 7,
    days_forward: int = 7,
    anchor_date: Optional[str] = None,
) -> dict:
    """根据描述获取今天的日期和近期日期范围。

    何时使用：
    - 用户问日期相关的问题时
    - 需要计算"未来三天"、"近一周"、"下周末"等模糊时间
    - 其他工具（如 get_travel_param）解析日期时需要锚点

    Args:
        days_back: 向前看的天数（包含今天），默认 7 天
        days_forward: 向后看的天数（包含今天），默认 7 天
        anchor_date: 锚点日期，"today" 或 "YYYY-MM-DD"，默认今天

    Returns:
        {
            "today": "2026-09-08",
            "today_weekday": "星期二",
            "recent_past": ["2026-09-02", ..., "2026-09-08"],   # 过去 N 天（含今天）
            "recent_future": ["2026-09-08", ..., "2026-09-15"],  # 未来 N 天（含今天）
            "date_range": {
                "start": "2026-09-02",
                "end": "2026-09-15"
            }
        }
    """
    base = _resolve_anchor_date(anchor_date)

    # 包含今天的过去 N 天（按时间正序）
    recent_past = [
        _format_date(base - timedelta(days=i))
        for i in range(days_back - 1, -1, -1)
    ]
    # 包含今天的未来 N 天（按时间正序）
    recent_future = [
        _format_date(base + timedelta(days=i))
        for i in range(0, days_forward)
    ]

    return {
        "today": _format_date(base),
        "today_weekday": _format_weekday_cn(base),
        "recent_past": recent_past,
        "recent_future": recent_future,
        "date_range": {
            "start": recent_past[0],
            "end": recent_future[-1],
        },
    }


@tool
def parse_relative_date(relative: str, anchor_date: Optional[str] = None) -> dict:
    """把相对时间（"未来三天"、"近一周"）解析成具体日期范围。

    何时使用：
    - get_travel_param 返回的 dates 是模糊的日期描述 时，需要转换
    - 用户说"明天"、"下周末"等相对时间时

    Args:
        relative: 相对时间描述，支持：
            - "今天"、"明天"、"后天"、"大后天"
            - "未来三天"、"未来一周"、"未来两周"、"未来一个月"
            - "近三天"、"近一周"、"近两周"
            - "下周"、"本周"、"上周"
            - "YYYY-MM-DD"（直接返回）
        anchor_date: 锚点日期，"today" 或 "YYYY-MM-DD"

    Returns:
        {
            "matched": "未来三天",
            "dates": ["2026-09-08", "2026-09-09", "2026-09-10"],
            "description": "今天 + 未来 2 天"
        }

    注意：
        - "未来 N 天" 包含今天 + 未来 N-1 天
        - "近 N 天" 包含今天 + 过去 N-1 天
        - 无法解析时返回 matched 但 dates 为空
    """
    base = _resolve_anchor_date(anchor_date)
    relative = relative.strip()

    # 1. 直接日期格式
    try:
        datetime.strptime(relative, "%Y-%m-%d")
        return {
            "matched": relative,
            "dates": [relative],
            "description": "具体日期",
        }
    except ValueError:
        pass

    # 2. 简单相对词
    simple_map = {
        "今天": 0,
        "明天": 1,
        "后天": 2,
        "大后天": 3,
    }
    if relative in simple_map:
        offset = simple_map[relative]
        target = base + timedelta(days=offset)
        return {
            "matched": relative,
            "dates": [_format_date(target)],
            "description": relative,
        }

    # 3. "未来/近 N 天/周/月"
    future_patterns = [
        ("未来一天", 1), ("未来两天", 2), ("未来三天", 3), ("未来四天", 4), ("未来五天", 5),
        ("未来六天", 6), ("未来七天", 7), ("未来一周", 7), ("未来两周", 14),
        ("未来半个月", 15), ("未来一个月", 30),
    ]
    past_patterns = [
        ("近一天", 1), ("近两天", 2), ("近三天", 3), ("近四天", 4), ("近五天", 5),
        ("近一周", 7), ("近两周", 14), ("近半个月", 15), ("近一个月", 30),
        ("昨天", -1), ("前天", -2),
    ]

    for pattern, n in future_patterns:
        if relative == pattern:
            dates = [_format_date(base + timedelta(days=i)) for i in range(n)]
            return {
                "matched": relative,
                "dates": dates,
                "description": f"今天 + 未来 {n - 1} 天",
            }

    for pattern, n in past_patterns:
        if relative == pattern:
            if n < 0:
                # 昨天/前天
                target = base + timedelta(days=n)
                return {
                    "matched": relative,
                    "dates": [_format_date(target)],
                    "description": relative,
                }
            dates = [_format_date(base - timedelta(days=i)) for i in range(n - 1, -1, -1)]
            return {
                "matched": relative,
                "dates": dates,
                "description": f"过去 {n - 1} 天 + 今天",
            }

    # 4. 周相关
    week_map = {
        "本周": 0,    # 本周一到周日
        "这周": 0,
        "下周": 1,    # 下周一到周日
        "上周": -1,   # 上周一到周日
    }
    if relative in week_map:
        offset_weeks = week_map[relative]
        # 周一
        monday = base - timedelta(days=base.weekday()) + timedelta(weeks=offset_weeks)
        dates = [_format_date(monday + timedelta(days=i)) for i in range(7)]
        return {
            "matched": relative,
            "dates": dates,
            "description": relative + "（周一到周日）",
        }

    # 5. 解析失败
    return {
        "matched": relative,
        "dates": [],
        "description": f"无法解析 '{relative}'",
    }