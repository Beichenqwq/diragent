"""校验 usage 对应的请求和回复是否仍与当前上下文一致。"""

from collections.abc import Sequence
from hashlib import sha256
from json import dumps

from agent.messages import AgentMessage, AssistantMessage
from agent.tools import AgentTool


def _fingerprint(value: object) -> str:
    text = dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(text.encode("utf-8")).hexdigest()


def request_context_fingerprint(
    *,
    model: str,
    system: str,
    messages: Sequence[AgentMessage],
    tools: Sequence[AgentTool],
) -> str:
    """记录请求输入；忽略本地 usage、校验信息和工具执行函数。"""
    return _fingerprint({
        "model": model,
        "system": system,
        "messages": [
            message.model_dump(
                mode="json", by_alias=False,
                exclude={"usage", "usage_context", "ok"},
            )
            for message in messages
        ],
        "tools": [
            {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
            for tool in tools
        ],
    })


def response_context_fingerprint(
    request_fingerprint: str,
    message: AssistantMessage,
) -> str:
    """把请求与最终回复绑定；修改回复内容或用量后校验不再匹配。"""
    return _fingerprint({
        "request": request_fingerprint,
        "response": message.model_dump(
            mode="json", by_alias=False, exclude={"usage_context"},
        ),
    })
