"""
健壮 JSON 解析工具
LLM 输出经常带 ```json 包裹、emoji、多余前后缀,
本工具尽力提取最大的 JSON 对象。
"""
import json
import re
from typing import Optional, Any


_JSON_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_json(text: str) -> Optional[Any]:
    """
    从任意 LLM 输出文本中提取 JSON 对象
    
    策略(按尝试顺序):
      1. 直接 json.loads
      2. ```json ... ``` 代码块抽取
      3. 找第一个 { 到匹配的 } 子串(支持嵌套)
      4. 失败返回 None
    """
    if not text:
        return None
    text = text.strip()
    
    # 1. 直接尝试
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    
    # 2. 代码块
    m = _JSON_CODE_BLOCK_RE.search(text)
    if m:
        block = m.group(1).strip()
        try:
            return json.loads(block)
        except (json.JSONDecodeError, ValueError):
            text = block  # 用 block 内容继续后续尝试
    
    # 3. 平衡花括号扫描(跳过字符串里的花括号)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    return None
    return None
