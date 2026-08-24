import re
from pathlib import Path
from typing import List

def clean_text(text: str) -> str:
    """清理文本：去除多余换行，替换为空格"""
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            if line.startswith('#'):
                # 标题保留原样
                cleaned_lines.append(line)
            else:
                # 替换多个空格为单个空格
                line = re.sub(r'\s+', ' ', line)
                cleaned_lines.append(line)
    # 合并，清理多余空格
    result = ' '.join(cleaned_lines)
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def split_paragraphs(content: str, length : int = 500) -> List[str]:
    """按段落分割 md 内容
    分割规则：
    1. 以 ## 开头的新标题作为新段落
    2. 连续的空行作为段落分隔
    """
    # 按标题分割
    parts = re.split(r'(?=^## .+$)', content, flags=re.MULTILINE)
    paragraphs = []
    for part in parts:
        # 每个部分内部按空行分割
        sub_paragraphs = re.split(r'\n\s*\n', part)
        for sub in sub_paragraphs:
            if sub.strip():
                ss = sub.strip()[:length] if len(sub.strip()) > length else sub.strip()
                paragraphs.append(ss)
    return paragraphs
