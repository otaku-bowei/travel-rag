
"""
环境变量或数据库配置读取服务
"""
import os
import string


def read_environment_config(key: string) -> string:
    return  os.environ.get(key)