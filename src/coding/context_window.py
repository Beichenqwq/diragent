"""上下文用量的分项估算和结果模型。"""

from collections.abc import Sequence
from dataclasses import dataclass
from json import dumps

from agent.context import request_context_fingerprint, response_context_fingerprint
from agent.messages import AgentMessage, AssistantMessage, ToolResultMessage
from agent.tools import AgentTool

CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 4
TOOL_OVERHEAD_TOKENS = 16
DEFAULT_CONTEXT_WINDOW_TOKENS = 128_000
DEFAULT_COMPACTION_RESERVE_TOKENS = 16_384
DEFAULT_COMPACTION_KEEP_RECENT_TOKENS = 20_000

@dataclass(frozen=True, slots=True)
class ContextUsageEstimate:
    """保存一次上下文估算的快照，数值由调用方计算并传入。"""

    total_tokens: int
    system_tokens: int
    message_tokens: int
    tool_tokens: int
    message_count: int
    provider_tokens: int = 0

    @property
    def uses_provider_usage(self) -> bool:
        """是否使用了 API 返回的用量来预测。"""
        return self.provider_tokens > 0

    @property
    def estimated_tokens(self) -> int:
        """总量中通过字符估算得到的部分。"""
        return self.total_tokens - self.provider_tokens


def auto_compaction_threshold_for_context_window(context_window_tokens: int) -> int | None:
    """为正数模型窗口预留摘要空间，非正数窗口不提供自动压缩阈值。"""
    if context_window_tokens <= 0:
        return None
    return max(1, context_window_tokens - DEFAULT_COMPACTION_RESERVE_TOKENS)


def estimate_text_tokens(text: str) -> int:
    """按每 4 个字符约 1 token 向上取整；空文本返回 0。"""
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def estimate_message_tokens(message: AgentMessage) -> int: # 处理单条消息，每条消息会加一个MESSAGE_OVERHEAD_TOKENS作为每条消息除了正文以外的额外开销。
    """估算消息正文及工具调用信息，包含固定消息开销。"""
    tokens = MESSAGE_OVERHEAD_TOKENS + estimate_text_tokens(message.content) # 正文消息
    if isinstance(message, AssistantMessage):
        tokens += sum( # 消息中包含工具调用
            estimate_text_tokens(call.name)
            + estimate_text_tokens(dumps(call.arguments, ensure_ascii=False))
            for call in message.tool_calls
        )
    elif isinstance(message, ToolResultMessage):
        tokens += estimate_text_tokens(message.name) # 工具结果消息里面还有一项name。
    return tokens


def estimate_tool_tokens(tool: AgentTool) -> int:
    """估算工具名称、描述和 JSON schema，包含固定定义开销。"""
    return (
        TOOL_OVERHEAD_TOKENS
        + estimate_text_tokens(tool.name)
        + estimate_text_tokens(tool.description)
        + estimate_text_tokens(dumps(tool.input_schema, ensure_ascii=False))
    )


def _last_applicable_provider_usage(
    *,
    model: str,
    system: str,
    messages: Sequence[AgentMessage],
    tools: Sequence[AgentTool],
) -> tuple[int, int] | None:
    """从最近回复往前查找 usage；只接受与当前请求输入和回复匹配的记录。"""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, AssistantMessage):
            continue
        if message.usage is None or not message.usage_context or message.usage.total_tokens <= 0:
            continue
        request_fingerprint = request_context_fingerprint(
            model=model, system=system, messages=messages[:index], tools=tools,
        )
        if message.usage_context == response_context_fingerprint(request_fingerprint, message):
            return index, message.usage.total_tokens
    return None


def estimate_context_usage(
    *,
    system: str,
    messages: Sequence[AgentMessage],
    tools: Sequence[AgentTool],
    model: str | None = None,
) -> ContextUsageEstimate:
    """预测当前上下文；有匹配 usage 时优先使用，否则整体估算。"""
    # 三个分项始终保留整个上下文的字符估算，API 没有提供这些真实分项。
    system_tokens = estimate_text_tokens(system)
    message_tokens = sum(estimate_message_tokens(message) for message in messages)
    tool_tokens = sum(estimate_tool_tokens(tool) for tool in tools)
    total_tokens = system_tokens + message_tokens + tool_tokens
    provider_tokens = 0

    if model is not None:
        usage_info = _last_applicable_provider_usage(
            model=model, system=system, messages=messages, tools=tools,
        )
        if usage_info is not None:
            usage_index, provider_tokens = usage_info
            total_tokens = provider_tokens + sum(
                estimate_message_tokens(message) for message in messages[usage_index + 1:]
            )

    return ContextUsageEstimate(
        total_tokens=total_tokens,
        system_tokens=system_tokens,
        message_tokens=message_tokens,
        tool_tokens=tool_tokens,
        message_count=len(messages),
        provider_tokens=provider_tokens,
    )
