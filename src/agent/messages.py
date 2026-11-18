"""定义一些agent transcript需要记录的消息的数据结构"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.types import JSONValue


## phase 1
def _to_camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.title() for part in parts)


class WireModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
        alias_generator=_to_camel,
    )


class UserMessage(WireModel):
    """存储用户发出的user消息，包含用户输入的文本"""

    role: Literal["user"] = "user" # 冒号是python的类型注解语法，写法是"变量名 + : + 类型"
    content: str


class ToolCall(WireModel):
    """assistant对工具调用的一次请求"""

    id: str # 工具调用的id
    name: str # 工具名
    arguments: dict[str, JSONValue] = Field(default_factory=dict) # 工具参数。Field:如果没有提供这个字段，就用dict创建一个空字典作为默认值。


class Usage(WireModel):
    """单次 API 请求的用量；input 包含缓存输入，cached 是其中的子集。"""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)


class AssistantMessage(WireModel):
    """存储模型输出的assistant消息，包含text和可能的工具调用"""

    role: Literal["assistant"] = "assistant"
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage | None = None
    # 用于确认 usage 对应的请求和回复仍与当前上下文一致；只在本地保存。
    usage_context: str | None = None


class ToolResultMessage(WireModel):
    """存储一次特定工具调用的结果"""

    role: Literal["tool"] = "tool"
    tool_call_id: str # 工具调用id
    name: str # 工具名
    content: str # 工具调用结果返回的内容
    ok: bool = True # 标记调用成功或失败


type AgentMessage = UserMessage | AssistantMessage | ToolResultMessage # 作为union类型，使得程序能够统一接收这些类型组成的列表list[AgentMessage]，而无需关心其中每一项是来自user还是assistant还是tool result。


if __name__ == "__main__":
    user_message = UserMessage(role="user", content="Read README.md")
    print(user_message)
    print(user_message.model_dump(), type(user_message.model_dump())) # dict
    print(user_message.model_dump_json(), type(user_message.model_dump_json())) # str
    pass
