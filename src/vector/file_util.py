import re
from pathlib import Path
from typing import List, Optional, Dict

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


def split_paragraphs(content: str, length: Optional[int] = None) -> List[str]:
    """按 --- 分割段落

    Args:
        content: md 文件内容（段落间用 --- 分隔）
        length: 段落最大长度，None 不截断（保留完整段落）

    Returns:
        段落列表（非空，已 strip）
    """
    parts = re.split(r'\n---\n', content)
    paragraphs = []
    for p in parts:
        s = p.strip()
        if not s:
            continue
        if length is not None and len(s) > length:
            s = s[:length]
        paragraphs.append(s)
    return paragraphs


def parse_md_paragraph(para: str) -> Dict[str, str]:
    """按行号解析 md 段落为结构化 dict

    格式（按 --- 分割后每段 5 行）：
      第 1 行：标题
      第 2 行：来源
      第 3 行：点赞 | 收藏
      第 4 行：链接
      第 5 行起：正文（合并所有剩余行）
    """
    info = {
        "title": "", "source": "", "likes": "",
        "favorites": "", "url": "", "content": "",
    }
    lines = [l.strip() for l in para.splitlines() if l.strip()]
    if len(lines) < 1:
        return info

    info["title"] = lines[0]
    if len(lines) < 2:
        return info
    info["source"] = lines[1]
    if len(lines) < 3:
        return info

    # 第 3 行：点赞: x | 收藏: y
    for part in lines[2].split("|"):
        p = part.strip()
        if p.startswith("点赞:"):
            info["likes"] = p.split(":", 1)[1].strip()
        elif p.startswith("收藏:"):
            info["favorites"] = p.split(":", 1)[1].strip()

    if len(lines) < 4:
        return info
    info["url"] = lines[3]

    if len(lines) < 5:
        return info
    # 第 5 行起：正文（合并）
    info["content"] = " ".join(lines[4:])
    return info
