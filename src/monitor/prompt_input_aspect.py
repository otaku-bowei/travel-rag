import aspectlib


# TODO--做到一些持久化层内，提供后面做数据分析
@aspectlib.Aspect
def log_llm_last_input_prompt():
    """打印 LLM 输入消息的切面"""
    yield aspectlib.Around
    method = yield aspectlib.Around
    args, kwargs = yield aspectlib.Arguments
    # 提取 messages 参数（第二个参数）
    if len(args) > 1:
        messages = args[1]
        print(f"\n📤 [{method.__name__}] Messages:")
        for msg in messages:
            if hasattr(msg, 'content'):
                print(f"  [{type(msg).__name__}]: {msg.content[:200]}...")
            else:
                print(f"  - {msg}")
    # 执行原方法
    result = yield aspectlib.Proceed
    yield aspectlib.Return(result)