"""会话中保存的消息记录。"""

from time import time
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent.messages import AgentMessage, Usage


class MessageEntry(BaseModel):
    """包装一条用户消息、助手消息或工具结果。"""

    model_config = ConfigDict(extra="forbid")

    type: Literal["message"] = "message"
    id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: float = Field(default_factory=time) # Field(default_factory=)创建对象时，如果没有传入这个字段，就调用指定函数，生成默认值
    message: Annotated[AgentMessage, Field(discriminator="role")] # 类型仍然是AgentMessage，后面是对这个类型的说明，这个说明可以是任意写的，但是Field(discriminator=)会被pydantic用来区分联合类型下的不同类型，用区分出来的类型再做校验，就不会用所有类型来做校验了。


class SessionInfoEntry(BaseModel):
    """创建会话时保存一次的元信息；cwd 仅记录路径。"""

    model_config = ConfigDict(extra="forbid")

    type: Literal["session_info"] = "session_info"
    session_id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: float = Field(default_factory=time)
    cwd: str


class CompactionEntry(BaseModel):
    """保存摘要和保留起点；usage 只记录生成该摘要的请求用量。"""

    model_config = ConfigDict(extra="forbid")

    type: Literal["compaction"] = "compaction"
    id: str = Field(default_factory=lambda: uuid4().hex)
    timestamp: float = Field(default_factory=time)
    summary: str
    first_kept_entry_id: str | None = None
    tokens_before: int | None = Field(default=None, ge=0)
    usage: Usage | None = None


type SessionEntry = MessageEntry | SessionInfoEntry | CompactionEntry
