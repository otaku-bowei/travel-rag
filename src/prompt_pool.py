"""
Few-shot 示例池 - 旅游问答
根据问题类型自动选择示例
"""
from typing import Optional


# Few-shot 示例池
EXAMPLE_POOL = {
    # 交通类
    "交通": [
        """例子1:
问题: 大阪到京都怎么走？
回答: 可以乘坐JR京都线，约15分钟，票价570日元。或者乘坐京阪电铁，约45分钟。

例子2:
问题: 从大阪环球影城到心斋桥怎么走？
回答: 乘坐JR环状线到大阪站，换乘御堂筋线到心斋桥站，约30分钟。

请回答用户问题：""",
    ],
    
    # 美食类
    "美食": [
        """例子1:
问题: 大阪有什么好吃的？
回答: 推荐道顿掘的章鱼烧、甲贺流章鱼烧，还有蟹道乐的螃蟹料理。

例子2:
问题: 京都美食推荐？
回答: 推荐锦市场逛吃、弘烤肉、菊乃井怀石料理。

请回答用户问题：""",
    ],
    
    # 景点类
    "景点": [
        """例子1:
问题: 大阪必去的景点？
回答: 环球影城、道顿掘、心斋桥、通天阁、阿倍野展望台。

例子2:
问题: 京都必去的景点？
回答: 清水寺、伏见稻荷大社、岚山、金阁寺、八坂神社。

请回答用户问题：""",
    ],
    
    # 行程规划
    "行程": [
        """例子1:
问题: 大阪三日游怎么安排？
回答: 
Day1: 环球影城
Day2: 道顿掘 + 心斋桥购物
Day3: 大阪城 + 通天阁

例子2:
问题: 京都两日游怎么安排？
回答:
Day1: 清水寺 + 二年坂 + 八坂神社
Day2: 伏见稻荷 + 岚山

请回答用户问题：""",
    ],
}


def select_example(user_question: str) -> str:
    """根据问题关键词选择合适的示例"""
    question_lower = user_question.lower()
    
    # 关键词匹配
    if any(kw in question_lower for kw in ["怎么走", "交通", "路线", "地铁", "jr", "新干线"]):
        return EXAMPLE_POOL["交通"]
    
    if any(kw in question_lower for kw in ["美食", "好吃", "餐厅", "料理", "吃什么", "食物"]):
        return EXAMPLE_POOL["美食"]
    
    if any(kw in question_lower for kw in ["景点", "必去", "观光", "旅游", "好玩"]):
        return EXAMPLE_POOL["景点"]
    
    if any(kw in question_lower for kw in ["安排", "日程", "几天", "行程", "计划", "路线"]):
        return EXAMPLE_POOL["行程"]
    
    # 默认返回空（使用 Zero-shot）
    return ""


def build_fewshot_prompt(user_question: str, use_fewshot: bool = True) -> str:
    """构建 Few-shot 提示词"""
    if not use_fewshot:
        return user_question
    
    example = select_example(user_question)
    if not example:
        return user_question
    
    return f"{example}\n{user_question}"


# 测试
if __name__ == "__main__":
    test_questions = [
        "大阪到京都交通怎么走？",
        "大阪有什么美食推荐？",
        "大阪三日游怎么安排？",
    ]
    
    for q in test_questions:
        print(f"\n问题: {q}")
        print(f"选用示例: {select_example(q)[:50]}...")