# DirAgent

> An agent that lives in your working directory.

一个 Coding Agent。用户通过终端 TUI 或 Web UI 提交任务，Agent 调用模型进行推理，使用本地工具读取、修改文件和执行命令，并保存会话记录。

## 安装与运行

需要 Python 3.12+ 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。在仓库根目录运行 `uv sync --locked` 来安装项目依赖。

并创建 `.env`，写入以下内容：

```dotenv
OPENAI_API_KEY=your-api-key
# OPENAI_BASE_URL=https://api.openai.com/v1
```

将 `YOUR_MODEL` 换成要使用的模型 ID，可任选一个入口运行本项目。

- TUI：`uv run python -m coding.tui_app --model YOUR_MODEL`
- Web：`uv run python -m coding.webui --model YOUR_MODEL`，打开 `http://127.0.0.1:8765`

## 架构图

下图展示当前代码中的模块关系。实线表示调用或数据读写，虚线表示事件回传或辅助模型请求。

```mermaid
flowchart TB
    User["用户"]

    subgraph Entry["交互入口"]
        subgraph UIs[" "]
            TUI["终端 TUI · Textual<br/>coding/tui_app.py"]
            Web["Web UI · FastAPI + SSE<br/>coding/webui.py + webui.html"]
        end
    end
    style UIs fill:none,stroke:#64748b,stroke-dasharray:5 4

    subgraph Coding["编程会话层 · src/coding/"]
        Session["CodingSession · session.py<br/>组装依赖、编排运行、保存消息"]
        Resources["技能与系统提示词<br/>skills.py / system_prompt.py"]
        Context["上下文管理<br/>context_window.py + session.py<br/>用量估算、摘要压缩、溢出恢复"]
        Memory["长期记忆<br/>memory_retrieval.py / memory.py<br/>memory_consolidation.py<br/>检索、提取、去重与整合"]
        Tools["本地工具 · tools.py<br/>read / write / edit / bash"]
    end

    subgraph Agent["通用 Agent 层 · src/agent/"]
        Harness["AgentHarness · harness.py<br/>持有 messages，接收 prompt / continue"]
        Loop["run_agent_loop · loop.py<br/>请求模型、执行工具、推进轮次"]
        State["SessionState · session/memory.py<br/>回放 entries，重建有效 messages"]
        Storage["JsonlSessionStorage · session/storage.py<br/>从 JSONL 读写会话 entries"]
    end

    subgraph AI["模型适配层 · src/ai/"]
        Provider["ModelProvider 接口<br/>OpenAICompatibleProvider<br/>实现 SSE 流式解析"]
    end

    API["OpenAI 兼容模型 API<br/>/chat/completions"]

    User --> UIs
    UIs --> Session

    Session --> Resources
    Session --> Context
    Session --> Memory
    Context -.->|请求摘要| Provider
    Memory -.->|检索、提取与整合请求| Provider

    Session -->|运行 Agent / 更新上下文| Harness
    Harness --> Loop
    Loop -->|stream_response| Provider
    Provider <-->|HTTP| API
    Loop -->|执行 tool_calls| Tools
    Tools -->|工具结果加入上下文| Loop

    Loop -.->|AgentEvent| Harness
    Harness -.->|AgentEvent| Session
    Session -.->|事件流| UIs

    Session -->|追加消息与压缩记录| Storage
    Storage -->|读取 entries| State
    State -->|恢复上下文状态| Session
```

### 核心流程

1. **加载会话**：`CodingSession.load()` 读取 JSONL，通过 `SessionState` 回放记录，加载技能、系统提示词和本地工具，再创建 `AgentHarness`。
2. **准备上下文**：`CodingSession.prompt()` 按需展开指定技能，注入记忆索引和检索到的记忆正文，并检查自动压缩阈值。
3. **运行 Agent**：`AgentHarness` 追加用户消息，调用 `run_agent_loop()`。循环通过 `ModelProvider` 请求模型；模型返回工具调用时，按顺序执行工具，将结果加入上下文后继续请求，直到获得最终回复、发生终止错误或达到轮次上限。
4. **显示与保存**：运行期间，`AgentEvent` 经由 Session 回传给界面；循环结束后，Session 追加保存本次新增消息。成功运行后，再提取长期记忆，并在满足阈值时整合已有记忆。
5. **压缩与恢复**：压缩将较早的消息生成摘要，追加 `CompactionEntry`，再回放为“摘要 + 近期消息”并替换 Harness 上下文；原始记录继续保留在 JSONL 中。`prompt()` 遇到上下文溢出时，会尝试压缩并继续运行一次。

## 上下文 & 持久化记录的维护

箭头指向操作写入的对象。`entries` 是 JSONL 中的会话记录，`messages` 是 `AgentHarness` 持有的运行时对话消息。

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 18, "rankSpacing": 82}}}%%
flowchart TB
    Save["保存新增消息<br/>_persist_new_messages()<br/>coding/session.py"]
    Load["加载会话<br/>CodingSession.load()<br/>coding/session.py"]
    Compact["压缩上下文<br/>_compact_saved_context()<br/>coding/session.py"]
    Prompt["接收用户输入<br/>AgentHarness.prompt()<br/>agent/harness.py"]
    Loop["处理回复和工具结果<br/>run_agent_loop()<br/>agent/loop.py"]

    Entries[("entries · JSONL 会话记录<br/>JsonlSessionStorage<br/>agent/session/storage.py")]
    Messages[["▣ messages · 运行时对话消息<br/>由 AgentHarness 持有<br/>agent/harness.py"]]

    Save -->|"追加 MessageEntry"| Entries
    Load -->|"空会话时追加<br/>SessionInfoEntry"| Entries
    Compact -->|"追加 CompactionEntry"| Entries

    Load -->|"用回放结果初始化"| Messages
    Compact -->|"替换为摘要和保留消息"| Messages
    Prompt -->|"追加 UserMessage"| Messages
    Loop -->|"追加 AssistantMessage<br/>和 ToolResultMessage"| Messages
```

记录加载和上下文压缩都会通过 `SessionState.from_entries()` 从 entries 回放出有效的 messages。压缩后追加的 `CompactionEntry` 用来保存压缩摘要和messages保留边界，原有 entries 继续留在 JSONL 中。

## 一次运行的事件流

下图展示正常运行的事件流。`ProviderEvent` 由 Provider 产出，`AgentEvent` 由 Agent 循环产出。右侧“AgentEvent转发链”代表 `AgentHarness → CodingSession → TUI 或 Web 后端` 的合并的事件转发链。每个发往右侧的 `AgentEvent` 都沿这条链传递。用户界面会收到主循环的 `AgentEvent` 并消费事件来向用户显示内容。`agent_end` 表示一次 Agent 循环的结束。

```mermaid
%%{init: {"sequence": {"mirrorActors": false}}}%%
sequenceDiagram
    participant API as 模型 API
    participant P as Provider
    participant L as Loop
    participant U as AgentEvent 转发链

    L-->>U: agent_start

    loop 逐轮推理，直到获得最终回复
        L-->>U: turn_start
        L->>P: stream_response
        P-->>L: response_start
        L-->>U: message_start
        P->>API: POST /chat/completions
        loop 模型响应中有文本增量时
            API-->>P: SSE 文本分片
            P-->>L: text_delta
            L-->>U: message_delta
        end
        opt 回复包含工具调用
            P-->>L: tool_call（每个调用一次）
            Note over P,L: Loop 将从 response_end 的完整消息中读取工具调用
        end
        P-->>L: response_end（AssistantMessage）
        L-->>U: message_end
        opt 助手消息包含工具调用
            loop 按顺序处理每个工具调用
                L-->>U: tool_execution_start
                L->>L: 执行工具，将结果加入上下文
                L-->>U: tool_execution_end
            end
        end
        L-->>U: turn_end
    end
    L-->>U: agent_end
```

## 致谢

DirAgent 在开发过程中大量参考了 Hugging Face 的 [Tau](https://github.com/huggingface/tau)。感谢 Tau 团队的开源。
