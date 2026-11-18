from typing import AsyncIterator, Any
from json import dumps, loads, JSONDecodeError # dumps返回字符串，dump直接写进文件

import httpx

from ai.env import OpenAICompatibleConfig
from ai.events import ProviderEvent, ProviderResponseStartEvent, ProviderErrorEvent, ProviderTextDeltaEvent, ProviderToolCallEvent, ProviderResponseEndEvent
from ai.provider import CancellationToken
from agent.context import request_context_fingerprint, response_context_fingerprint
from agent.messages import AgentMessage, UserMessage, AssistantMessage, ToolResultMessage, ToolCall, Usage
from agent.tools import AgentTool
from agent.types import JSONValue


class OpenAICompatibleProvider:
    def __init__(
        self,
        config: OpenAICompatibleConfig,
        *,
        client: httpx.AsyncClient | None = None
    ):
        self._config = config
        self._client = client
        self._owns_client = client is None

    def stream_response(
        self,
        *,
        model: str,
        system: str,
        messages: list[AgentMessage],
        tools: list[AgentTool],
        signal: CancellationToken | None = None
    ) -> AsyncIterator[ProviderEvent]:
        """实现provider的流式传输接口"""

        async def iterator() -> AsyncIterator[ProviderEvent]:
            client = self._get_client()
            payload = _build_chat_payload(
                model=model,
                system=system,
                messages=messages,
                tools=tools
            )
            request_fingerprint = request_context_fingerprint(
                model=model, system=system, messages=messages, tools=tools,
            )
            if self._config.include_usage:
                payload["stream_options"] = {"include_usage": True}
            if self._config.thinking is not None:
                payload["thinking"] = {"type": self._config.thinking}
            headers = {"Authorization": f"Bearer {self._config.api_key}"}
            url = f"{self._config.base_url}/chat/completions"

            yield ProviderResponseStartEvent(model=model)

            try:
                async with client.stream("POST", url=url, json=payload, headers=headers) as response:
                    if response.status_code >= 400:
                        body = await response.aread()
                        yield ProviderErrorEvent(
                            message=f"Provider request failed with status code {response.status_code}",
                            data={"body": body.decode(errors="replace")}
                        )
                        return

                    content_parts: list[str] = []
                    tool_call_builders: dict[int, _ToolCallBuilder] = {}
                    finish_reason: str | None = None
                    usage: Usage | None = None

                    async for line in response.aiter_lines():
                        if signal is not None and signal.is_cancelled():
                            return

                        event = _parse_sse_line(line)
                        if event is None:
                            continue
                        if event == "[DONE]":
                            break

                        chunk = _loads_object(event)
                        if chunk is None:
                            yield ProviderErrorEvent(message="Provider returned invalid JSON chunk")
                            return

                        # 用量 chunk 可以没有 choices，先保存整次请求的用量。
                        chunk_usage = _parse_chunk_usage(chunk)
                        if chunk_usage is not None:
                            usage = chunk_usage

                        choice = _first_choice(chunk)
                        if choice is None:
                            continue

                        finish_reason = choice.get("finish_reason")
                        delta = choice.get("delta")
                        if not isinstance(delta, dict):
                            continue

                        content = delta.get("content") # 处理content delta
                        if isinstance(content, str) and content:
                            content_parts.append(content)
                            yield ProviderTextDeltaEvent(delta=content)

                        tool_call_deltas = _tool_call_deltas(delta) # 处理tool_call delta
                        for tool_call_delta in tool_call_deltas:
                            index = tool_call_delta.get("index", 0)
                            builder = tool_call_builders.setdefault(index, _ToolCallBuilder(index))
                            builder.add_delta(tool_call_delta)

                    # 流式传输完毕
                    tool_calls = [
                        tool_call_builders[index].build() for index in sorted(tool_call_builders)
                    ]
                    for tool_call in tool_calls:
                        yield ProviderToolCallEvent(tool_call=tool_call)

                    message = AssistantMessage(
                        content="".join(content_parts),
                        tool_calls=tool_calls,
                        usage=usage,
                    )
                    if usage is not None:
                        message.usage_context = response_context_fingerprint(request_fingerprint, message)
                    yield ProviderResponseEndEvent(message=message, finish_reason=finish_reason)
            except httpx.HTTPError as exception:
                yield ProviderErrorEvent(message=str(exception))

        return iterator()

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None


def _build_chat_payload(
    *,
    model: str,
    system: str,
    messages: list[AgentMessage],
    tools: list[AgentTool]
) -> dict[str, JSONValue]:
    payload: dict[str, JSONValue] = { # 给模型api传的数据都是json字典类型
        "model": model,
        "stream": True,
        "messages": [
            _system_message(system),
            *[_message_to_openai(message) for message in messages]
        ]
    }
    if tools:
        payload["tools"] = [_tool_to_openai(tool) for tool in tools]
    
    return payload


def _system_message(system: str) -> dict[str, JSONValue]:
    """系统提示词"""
    return {"role": "system", "content": system}


def _message_to_openai(message: AgentMessage) -> dict[str, JSONValue]:
    """给compatibleAPI输入的以前的消息，包括user, assistant, toolResult工具执行结果"""
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.content}

    if isinstance(message, AssistantMessage):
        item: dict[str, JSONValue] = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            item["tool_calls"] = [
                _tool_call_to_openai(tool_call) for tool_call in message.tool_calls
            ]
        return item

    if isinstance(message, ToolResultMessage):
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "name": message.name,
            "content": message.content
        }


def _tool_call_to_openai(tool_call: ToolCall) -> dict[str, JSONValue]:
    """给compatibleAPI输入的以前assistant message里的工具调用信息"""
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": dumps(tool_call.arguments)
        },
    }


def _tool_to_openai(tool: AgentTool) -> dict[str, JSONValue]:
    """给compatibleAPI输入的工具定义"""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema
        }
    }


class _ToolCallBuilder:
    def __init__(self, index: int) -> None:
        self.id = ""
        self.name = ""
        self.arguments_parts: list[str] = []
        self.index = index

    def add_delta(self, delta: dict[str, Any]) -> None:
        id = delta.get("id")
        if isinstance(id, str) and id:
            self.id = id

        function = delta.get("function")
        if not isinstance(function, dict):
            return

        name = function.get("name")
        if isinstance(name, str) and name: # 兜底，防止模型流式的后续片段返回空的工具名称，覆盖初次的工具名称
            self.name = name

        arguments_parts = function.get("arguments")
        if isinstance(arguments_parts, str):
            self.arguments_parts.append(arguments_parts)

    def build(self) -> ToolCall:
        arguments_text = "".join(self.arguments_parts)
        arguments = _loads_object(arguments_text) if arguments_text else {}
        if arguments is None:
            arguments = {"_raw_arguments": arguments_text}
        return ToolCall(
            id=self.id,
            name=self.name,
            arguments=arguments
        )


def _loads_object(value: str) -> dict[str, JSONValue] | None:
    try:
        loaded = loads(value)
    except JSONDecodeError: 
        return None
    if isinstance(loaded, dict):
        return loaded
    return None


def _parse_sse_line(line: str) -> str | None:
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    return line.removeprefix("data:").strip()


def _parse_chunk_usage(chunk: dict[str, JSONValue]) -> Usage | None:
    """读取 OpenAI Chat Completions 用量；无有效数据时不伪造零用量。"""
    data = chunk.get("usage")
    if not isinstance(data, dict):
        return None

    input_tokens = data.get("prompt_tokens")
    output_tokens = data.get("completion_tokens")
    total_tokens = data.get("total_tokens")
    if any(type(value) is not int or value < 0
           for value in (input_tokens, output_tokens, total_tokens)):
        return None

    cached_input_tokens = None
    details = data.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
        if type(cached) is int and cached >= 0:
            cached_input_tokens = cached

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached_input_tokens,
    )


def _first_choice(chunk: dict[str, JSONValue]) -> dict[str, JSONValue] | None:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    return choice


def _tool_call_deltas(delta: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = delta.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [tool_call for tool_call in tool_calls if isinstance(tool_call, dict)]


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os

    load_dotenv()

    api_key = os.environ.get("DEEPSEEK_API_KEY")

    config = OpenAICompatibleConfig(api_key=api_key, base_url="https://api.deepseek.com")
    provider = OpenAICompatibleProvider(config=config)

    async def test():
        messages = [UserMessage(content="Say Hi"),]
        async for event in provider.stream_response(model="deepseek-chat",
                                system="",
                                messages=messages,
                                tools=[]):
            print(event)
        await provider.aclose()

    async def test_tool_call():
        from agent.tools import AgentToolResult

        async def add(arguments): # 工具的执行函数
            return AgentToolResult(
                tool_call_id="", name="add",
                content=str(arguments["a"] - arguments["b"]), ok=True,
            )

        tool = AgentTool( # 定义工具
            name="add",
            description="计算 a + b",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, # 如果type是integer，问题是小数的话(例如9.11+9.8)，可能会出现工具执行错误，但模型回答正确的情况。如果是很大的大数字(911888888888888888 + 98000007000000004)，工具执行给出错误的结果(工具改成相减)，模型也会回答这个工具错误的结果，他不进行自己纠错了。根本原因是，工具调用结果只是作为了模型的输入上下文。
                "required": ["a", "b"],
            },
            executor=add,
        )
        # 实验观察：
        # 1. 工具参数声明为 integer，但问题是 9.11 + 9.8 时，
        #    模型传入了 9 和 10，工具正确返回 19，模型最终却回答了 18.91。
        #    说明最终答案正确，不代表工具调用参数正确。
        # 2. 大整数加法实验中，故意把工具实现改为减法，
        #    模型沿用了工具返回的错误结果，没有在最终回答中纠正。
        # 机制：工具结果作为 role="tool" 消息进入模型上下文，
        # 并不直接决定最终答案；模型可能采用，也可能偏离它。
        # 这两次实验不能证明模型会稳定纠错，也不能确定差异是否由数字大小导致。
        messages: list[AgentMessage] = [UserMessage(content="911888888888888888 + 98000007000000004 等于多少？")]
        tool_results: list[ToolResultMessage] = []
        async for event in provider.stream_response(
            model="deepseek-chat",
            system="请调用工具计算。",
            messages=messages,
            tools=[tool],
        ):
            print(event)
            if isinstance(event, ProviderToolCallEvent):
                print("收到ProviderToolCallEvent事件，执行工具")
                result = await tool.execute(event.tool_call.arguments)
                print("工具结果：", result.content)
                tool_results.append(ToolResultMessage(
                    tool_call_id=event.tool_call.id,
                    name=event.tool_call.name,
                    content=result.content,
                    ok=result.ok,
                ))
            if isinstance(event, ProviderResponseEndEvent):
                # 先记录模型的工具调用请求，再记录对应的工具结果。
                messages.append(event.message)
                messages.extend(tool_results)

        if tool_results:
            async for event in provider.stream_response(
                model="deepseek-chat",
                system="请调用工具计算。",
                messages=messages,
                tools=[],
            ):
                print(event)
        await provider.aclose()

    import asyncio
    # asyncio.run(test())
    asyncio.run(test_tool_call())
    
