"""定义和工具本身相关的一些数据结构（工具定义、执行函数、模型请求、执行结果）"""

from collections.abc import Callable, Awaitable
from dataclasses import dataclass

from pydantic import Field

from agent.messages import WireModel
from agent.types import JSONValue

ToolExecutor = Callable[[dict[str, JSONValue]], Awaitable["AgentToolResult"]] # Callable[[], ]表示可调用的对象，里面第一项需要是一个列表，表示调用时传入的参数，
# 第二项表示返回的类型，这里返回的是Awaitable[]可等待对象，中括号里表示等待完成得到什么类型，加双引号表示前向引用，因为在这一行之前AgentToolResult还没有定义。
# 若executor是async def定义的，调用下面的异步执行函数executor会产生可等待对象（协程对象），等协程执行完才得到AgentToolResult。


class AgentToolResult(WireModel):
    """AgentTool执行结果的直接结构化输出，后续在loop中将会被转化为ToolResultMessage并append到transcript中"""

    tool_call_id: str
    name: str
    content: str
    ok: bool
    data: dict[str, JSONValue] | None = None # 工具结果的额外结构化信息
    error: str | None = None


@dataclass(frozen=True, slots=True) # dataclass能够给类中定义的字段自动创建一个__init__函数，frozen: 禁止给字段重新赋值；slots：将这些字段以更紧凑的方式存储（放到固定的槽位里面），不能再添加新的属性。
class AgentTool: # 因为会包含本地执行函数，不适合作为序列化的消息传递，所以不使用pydantic model。
    """描述一个agent可以提供给模型的工具"""

    name: str
    description: str
    input_schema: dict[str, JSONValue] 
    executor: ToolExecutor
    prompt_snippet: str | None = None
    prompt_guidelines: tuple[str, ...] = ()

    async def execute(self, arguments: dict[str, JSONValue]) -> AgentToolResult:
        return await self.executor(arguments)


