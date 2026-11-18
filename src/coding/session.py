"""编程会话的运行配置。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import date
import logging
from pathlib import Path
from json import dumps

from ai.events import ProviderErrorEvent, ProviderResponseEndEvent, ProviderTextDeltaEvent
from ai.provider import ModelProvider
from agent.events import AgentEvent, ErrorEvent
from agent.harness import AgentHarness, AgentHarnessConfig
from agent.messages import AgentMessage, AssistantMessage, ToolResultMessage, Usage, UserMessage
from agent.session.entries import CompactionEntry, MessageEntry, SessionInfoEntry
from agent.session.memory import SessionState
from agent.session.storage import JsonlSessionStorage
from agent.tools import AgentTool
from coding.context_window import (
    DEFAULT_COMPACTION_KEEP_RECENT_TOKENS,
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    ContextUsageEstimate,
    auto_compaction_threshold_for_context_window,
    estimate_context_usage,
    estimate_message_tokens,
)
from coding.memory import (
    build_memory_dialogue, build_memory_extraction_prompt, extract_memory_candidates,
    load_existing_memories, load_memory_index, should_store_memory, store_memory,
)
from coding.resources import ResourceError
from coding.memory_retrieval import retrieve_memories
from coding.memory_consolidation import (
    DEFAULT_MEMORY_CONSOLIDATION_THRESHOLD, consolidate_memory_candidates,
    load_memory_snapshot, rebuild_memories,
)
from coding.skills import Skill, expand_skill, load_skills
from coding.system_prompt import BuildSystemPromptOptions, ProjectContextFile, build_system_prompt
from coding.system_prompt import format_memories_for_prompt, format_memory_index_for_prompt
from coding.tools import create_coding_tools

COMPACTION_SUMMARY_PREFIX = "Previous conversation summary:\n"

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation "
    "between a user and an AI coding assistant, then produce a structured summary "
    "following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the "
    "conversation. ONLY output the structured summary."
)

SUMMARIZATION_PROMPT = (
    "The messages above are a conversation to summarize. Create a structured context "
    "checkpoint summary that another LLM will use to continue the work.\n\n"
    "Use this EXACT format:\n\n"
    "## Goal\n"
    "[What is the user trying to accomplish? Can be multiple items if the session "
    "covers different tasks.]\n\n"
    "## Constraints & Preferences\n"
    "- [Any constraints, preferences, or requirements mentioned by user]\n"
    '- [Or "(none)" if none were mentioned]\n\n'
    "## Progress\n"
    "### Done\n"
    "- [x] [Completed tasks/changes]\n\n"
    "### In Progress\n"
    "- [ ] [Current work]\n\n"
    "### Blocked\n"
    "- [Issues preventing progress, if any]\n\n"
    "## Key Decisions\n"
    "- **[Decision]**: [Brief rationale]\n\n"
    "## Next Steps\n"
    "1. [Ordered list of what should happen next]\n\n"
    "## Critical Context\n"
    "- [Any data, examples, or references needed to continue]\n"
    '- [Or "(none)" if not applicable]\n\n'
    "Keep each section concise. Preserve exact file paths, function names, and error "
    "messages."
)

UPDATE_SUMMARIZATION_PROMPT = (
    "The messages above are NEW conversation messages to incorporate into the existing "
    "summary provided in <previous-summary> tags.\n\n"
    "Update the existing structured summary with new information. RULES:\n"
    "- PRESERVE all existing information from the previous summary\n"
    "- ADD new progress, decisions, and context from the new messages\n"
    '- UPDATE the Progress section: move items from "In Progress" to "Done" when '
    "completed\n"
    '- UPDATE "Next Steps" based on what was accomplished\n'
    "- PRESERVE exact file paths, function names, and error messages\n"
    "- If something is no longer relevant, you may remove it\n\n"
    "Use this EXACT format:\n\n"
    "## Goal\n"
    "[Preserve existing goals, add new ones if the task expanded]\n\n"
    "## Constraints & Preferences\n"
    "- [Preserve existing, add new ones discovered]\n\n"
    "## Progress\n"
    "### Done\n"
    "- [x] [Include previously done items AND newly completed items]\n\n"
    "### In Progress\n"
    "- [ ] [Current work - update based on progress]\n\n"
    "### Blocked\n"
    "- [Current blockers - remove if resolved]\n\n"
    "## Key Decisions\n"
    "- **[Decision]**: [Brief rationale] (preserve all previous, add new)\n\n"
    "## Next Steps\n"
    "1. [Update based on current state]\n\n"
    "## Critical Context\n"
    "- [Preserve important context, add new if needed]\n\n"
    "Keep each section concise. Preserve exact file paths, function names, and error "
    "messages."
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CodingSessionConfig:
    """组合模型、存储和本地编程环境所需的配置。"""

    provider: ModelProvider
    model: str
    storage: JsonlSessionStorage
    cwd: Path
    system: str | None = None
    # None 表示使用默认 coding tools；空列表表示不提供工具。
    tools: list[AgentTool] | None = None
    context_files: Sequence[ProjectContextFile] = ()
    current_date: date | None = None
    extra_guidelines: Sequence[str] = ()
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    auto_compact_enabled: bool = True
    auto_compact_token_threshold: int | None = None
    max_turns: int = 8
    # 超过此条数才整合；0 关闭自动整合。
    memory_consolidation_threshold: int = DEFAULT_MEMORY_CONSOLIDATION_THRESHOLD


@dataclass(frozen=True, slots=True)
class CompactionPlan:
    """一次压缩使用的保留边界、待摘要消息和被替代的上下文行数。"""

    first_kept_entry_id: str
    replaced_entry_count: int
    messages_to_summarize: tuple[AgentMessage, ...]


@dataclass(frozen=True, slots=True)
class _GeneratedSummary:
    """模型生成的摘要及本次摘要请求用量，不表示会话累计用量。"""

    text: str
    usage: Usage | None


class CodingSession:
    """持有运行配置和从存储中恢复的会话状态。"""

    def __init__(
        self,
        config: CodingSessionConfig,
        *,
        state: SessionState,
        harness: AgentHarness,
        skills: Sequence[Skill],
    ) -> None:
        self._config = config
        self._state = state
        self._harness = harness
        self._base_system = harness.config.system
        self._skill_by_name = {skill.name: skill for skill in skills}
        self._busy = False

    @classmethod
    def load(cls, config: CodingSessionConfig) -> CodingSession:
        """同步读取记录；空会话写入元信息后重建状态。"""
        entries = config.storage.read_all()
        if not entries: # 需要新建session
            info = SessionInfoEntry(cwd=str(config.cwd.resolve()))
            config.storage.append(info)
            entries = [info]

        state = SessionState.from_entries(entries) # 创建SessionState
        tools = config.tools if config.tools is not None else create_coding_tools(config.cwd) # 如果没有在config中传入tools就创建默认的工具
        skills = load_skills(config.cwd)
        system = (
            config.system
            if config.system is not None
            else build_system_prompt(BuildSystemPromptOptions(
                cwd=config.cwd,
                tools=tools,
                skills=skills,
                context_files=config.context_files,
                current_date=config.current_date,
                extra_guidelines=config.extra_guidelines,
            ))
        )
        harness = AgentHarness(
            AgentHarnessConfig(
                provider=config.provider,
                model=config.model,
                system=system,
                tools=tools,
                max_turns=config.max_turns,
            ),
            messages=state.messages, # 恢复出来的历史消息
        )
        return cls(config, state=state, harness=harness, skills=skills)

    @property
    def config(self) -> CodingSessionConfig:
        """返回会话配置，供创建其他会话时复用模型和编程环境。"""
        return self._config

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        """返回 Harness 当前消息的元组快照；消息对象本身不复制。"""
        return self._harness.messages

    @property
    def context_usage(self) -> ContextUsageEstimate:
        """根据当前配置和消息预测上下文；优先使用匹配的 API usage。"""
        return estimate_context_usage(
            model=self._harness.config.model,
            system=self._harness.config.system,
            messages=self._harness.messages,
            tools=self._harness.config.tools,
        )

    @property
    def context_window_tokens(self) -> int:
        """返回调用方配置的模型窗口，未配置时使用默认值。"""
        return self._config.context_window_tokens

    @property
    def auto_compaction_enabled(self) -> bool:
        """返回自动压缩开关；不影响手动 compact()。"""
        return self._config.auto_compact_enabled

    @property
    def auto_compact_token_threshold(self) -> int | None:
        """关闭时无阈值，开启时显式值优先，否则由模型窗口计算。"""
        if not self.auto_compaction_enabled:
            return None
        if self._config.auto_compact_token_threshold is not None:
            return self._config.auto_compact_token_threshold
        return auto_compaction_threshold_for_context_window(self.context_window_tokens)

    @property
    def state(self) -> SessionState:
        """返回最近加载或保存后重建的状态；对象本身仍可修改。"""
        return self._state

    @property
    def storage(self) -> JsonlSessionStorage:
        """返回当前会话使用的存储对象。"""
        return self._config.storage

    def _recent_preserving_compaction_plan(self) -> CompactionPlan | None:
        """从已保存的有效上下文生成compaction plan；待摘要或保留部分为空时返回 None。"""
        def _first_recent_context_index(
            rows: tuple[tuple[str, AgentMessage], ...],
            *,
            keep_recent_tokens: int = DEFAULT_COMPACTION_KEEP_RECENT_TOKENS,
        ) -> int:
            """先按近期 token 预算定位候选，再优先调整到 user 或非工具结果消息。"""
            if keep_recent_tokens <= 0:
                return len(rows)

            accumulated_tokens = 0
            candidate_index: int | None = None
            for index in range(len(rows) - 1, -1, -1):
                _, message = rows[index]
                accumulated_tokens += estimate_message_tokens(message)
                if accumulated_tokens >= keep_recent_tokens:
                    candidate_index = index
                    break

            if candidate_index is None:
                return 0

            candidate_message = rows[candidate_index][1] # 候选起点规则1
            if candidate_message.role == "user" and candidate_index > 0:
                return candidate_index

            next_user_index = next( # 候选起点规则2
                (index for index in range(candidate_index + 1, len(rows))
                 if rows[index][1].role == "user"),
                None,
            )
            if next_user_index is not None:
                return next_user_index
            if candidate_message.role == "user":
                return 0 # 规则1兜底：执行到这里，说明排除了两种情况：1候选是user且下标>0，会在规则1返回；2候选不是user或候选是user但下标为0但候选后面有user，会在规则2返回。因此这种情况是：候选是user，且候选下标是0，且没有后续user。这种情况就返回0表示全部消息都保留

            for index in range(candidate_index, len(rows)): # 候选起点规则3
                if rows[index][1].role != "tool":
                    return index
            return len(rows) # 规则3兜底：候选自身及其后续消息全是工具结果，找不到非工具结果消息，就返回 `len(rows)`，表示保留部分为空。

        rows = tuple(zip(self._state.context_entry_ids, self._state.messages, strict=True))
        if len(rows) < 2:
            return None

        first_kept_index = _first_recent_context_index(rows)
        if first_kept_index <= 0 or first_kept_index >= len(rows): # 待摘要或保留部分为空
            return None

        replaced = rows[:first_kept_index]
        return CompactionPlan(
            first_kept_entry_id=rows[first_kept_index][0],
            replaced_entry_count=len(replaced),
            messages_to_summarize=tuple(message for _, message in replaced),
        )

    async def _generate_compaction_summary(
        self,
        messages: tuple[AgentMessage, ...],
        *,
        custom_instructions: str | None = None,
    ) -> _GeneratedSummary:
        """直接请求当前模型生成摘要，不将请求或回复加入会话历史。"""
        def build_compaction_summary_prompt(
            messages: tuple[AgentMessage, ...],
            *,
            custom_instructions: str | None = None,
        ) -> str:
            """首条为旧摘要时构建更新请求，否则构建首次摘要请求。"""
            previous_summary: str | None = None
            new_messages = messages
            if messages and isinstance(messages[0], UserMessage):
                first_text = messages[0].content
                if first_text.startswith(COMPACTION_SUMMARY_PREFIX):
                    previous_summary = first_text.removeprefix(COMPACTION_SUMMARY_PREFIX)
                    new_messages = messages[1:]

            lines: list[str] = []
            for index, message in enumerate(new_messages, start=1):
                attributes = f"index={index} role={message.role}"
                if isinstance(message, ToolResultMessage):
                    attributes += f" name={message.name} error={str(not message.ok).lower()}"
                lines.append(f"<message {attributes}>")
                if message.content:
                    lines.append(message.content)
                if isinstance(message, AssistantMessage) and message.tool_calls:
                    lines.append("<tool-calls>")
                    for call in message.tool_calls:
                        lines.append(f"- {call.name}: {dumps(call.arguments, ensure_ascii=False)}")
                    lines.append("</tool-calls>")
                lines.append("</message>")
            conversation = "\n".join(lines) if lines else "(no new messages)"
            prompt = f"<conversation>\n{conversation}\n</conversation>\n\n"
            base_prompt = (
                UPDATE_SUMMARIZATION_PROMPT if previous_summary is not None else SUMMARIZATION_PROMPT
            )
            if previous_summary is not None:
                prompt += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
            instructions = custom_instructions.strip() if custom_instructions is not None else ""
            if instructions:
                base_prompt = f"{base_prompt}\n\nAdditional focus: {instructions}"
            return f"{prompt}{base_prompt}"

        prompt = build_compaction_summary_prompt(messages, custom_instructions=custom_instructions)
        text_parts: list[str] = []
        final_text: str | None = None
        usage: Usage | None = None
        async for event in self._harness.config.provider.stream_response(
            model=self._harness.config.model,
            system=SUMMARIZATION_SYSTEM_PROMPT,
            messages=[UserMessage(content=prompt)],
            tools=[],
        ):
            if isinstance(event, ProviderTextDeltaEvent):
                text_parts.append(event.delta)
            elif isinstance(event, ProviderResponseEndEvent):
                final_text = event.message.content
                usage = event.message.usage
            elif isinstance(event, ProviderErrorEvent):
                raise RuntimeError(f"Compaction summarization failed: {event.message}")

        summary = (final_text if final_text is not None else "".join(text_parts)).strip()
        if not summary:
            raise RuntimeError("Compaction summarization returned an empty summary")
        return _GeneratedSummary(text=summary, usage=usage)

    async def compact(self, instructions: str | None = None) -> str:
        """手动压缩已保存上下文，追加记录并替换消息，成功返回摘要。"""
        if self._busy:
            raise RuntimeError("CodingSession is already running or compacting")
        self._busy = True
        try:
            summary = await self._compact_saved_context(instructions)
            if summary is None:
                raise ValueError("Not enough context to compact while preserving recent entries")
            return summary
        finally:
            self._busy = False

    async def _compact_saved_context(self, instructions: str | None = None) -> str | None:
        """从已保存的 entry 恢复有效 message，对较早的 message 生成摘要。

        追加 CompactionEntry，保留近期 message，并替换内存中的 message 列表。
        调用前须由外层将 _busy 设为 True。
        """
        entries_before = self._config.storage.read_all()
        persisted_state = SessionState.from_entries(entries_before)
        if self._state != persisted_state or self.messages != tuple(persisted_state.messages):
            raise RuntimeError("Cannot compact: context does not match saved session state")
        plan = self._recent_preserving_compaction_plan() # 生成compact plan
        if plan is None:
            return None
        tokens_before = self.context_usage.total_tokens
        generated = await self._generate_compaction_summary( # 请求摘要
            plan.messages_to_summarize, custom_instructions=instructions,
        )
        # 请求模型期间可能等待较久，保存前再次确认规划依据没有被外部修改。
        if (self._state != persisted_state
                or self.messages != tuple(persisted_state.messages)
                or self._config.storage.read_all() != entries_before):
            raise RuntimeError("Cannot compact: session context changed during summarization")
        if (not plan.first_kept_entry_id
                or plan.first_kept_entry_id not in self._state.context_entry_ids):
            raise ValueError("Compaction boundary must identify a current context entry")
        entry = CompactionEntry(
            summary=generated.text, first_kept_entry_id=plan.first_kept_entry_id,
            tokens_before=tokens_before, usage=generated.usage,
        )
        self._config.storage.append(entry)
        self._refresh_persisted_state()
        self._harness.replace_messages(self._state.messages)
        return generated.text

    async def _try_auto_compact(self) -> bool:
        """超过阈值时尝试压缩；失败后磁盘和内存仍一致则继续普通运行。"""
        threshold = self.auto_compact_token_threshold
        if threshold is None or threshold <= 0 or self.context_usage.total_tokens <= threshold:
            return False
        try:
            return await self._compact_saved_context() is not None
        except Exception as exc:
            logger.warning("Automatic compaction failed: %s", exc)
            if (SessionState.from_entries(self._config.storage.read_all()) != self._state
                    or self.messages != tuple(self._state.messages)):
                raise
            return False

    async def _try_overflow_compact(self) -> bool:
        """溢出恢复忽略自动开关和阈值；摘要失败时保留原错误。"""
        try:
            return await self._compact_saved_context() is not None
        except Exception as exc:
            logger.warning("Overflow compaction failed: %s", exc)
            if (SessionState.from_entries(self._config.storage.read_all()) != self._state
                    or self.messages != tuple(self._state.messages)):
                raise
            return False

    def _refresh_persisted_state(self) -> None:
        """成功读取和回放全部记录之后，才更新 state。"""
        self._state = SessionState.from_entries(self._config.storage.read_all())

    async def prompt( # 在一次调用实际运行前压缩上下文
        self, content: str, *, skill_name: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """可选地展开指定技能，再运行并保存；未知技能在修改消息前报错。"""
        if self._busy:
            raise RuntimeError("CodingSession is already running or compacting")
        self._busy = True
        try:
            current_prompt = content
            if skill_name is not None:
                skill = self._skill_by_name.get(skill_name)
                if skill is None:
                    raise ResourceError(f"Unknown skill: {skill_name}")
                content = expand_skill(skill, content)

            await self._prepare_memory_context(current_prompt)
            await self._try_auto_compact()
            before_count = len(self._harness.messages)
            overflow = False
            failed = False
            async for event in self._harness.prompt(content):
                if isinstance(event, ErrorEvent):
                    failed = True
                    overflow = overflow or is_context_overflow_error(event)
                yield event
            self._persist_new_messages(before_count)
            if overflow:
                if not await self._try_overflow_compact():
                    return
                # 原输入已经保存并参与压缩，不再 prompt；本次只重试一次。
                before_count = len(self._harness.messages)
                failed = False
                async for event in self._harness.continue_():
                    if isinstance(event, ErrorEvent):
                        failed = True
                    yield event
                self._persist_new_messages(before_count)
            if not failed:
                await self._try_extract_memories(self._harness.messages)
        finally:
            self._busy = False

    async def continue_(self) -> AsyncIterator[AgentEvent]:
        """从已有上下文继续运行，不新增用户输入；结束后保存消息。"""
        if self._busy:
            raise RuntimeError("CodingSession is already running or compacting")
        self._busy = True
        try:
            before_count = len(self._harness.messages) # before_count: 本次运行开始前，agentharness中已有的消息数量
            failed = False
            async for event in self._harness.continue_():
                if isinstance(event, ErrorEvent):
                    failed = True
                yield event
            self._persist_new_messages(before_count)
            if not failed:
                await self._try_extract_memories(self._harness.messages)
        finally:
            self._busy = False

    async def _prepare_memory_context(self, content: str) -> None:
        """注入完整索引和本轮选中正文；检索失败仍保留已加载的索引。"""
        self._harness.config.system = self._base_system
        try:
            index = await asyncio.to_thread(load_memory_index, self._config.cwd)
            self._harness.config.system += format_memory_index_for_prompt(index)
            memories = await retrieve_memories(
                self._harness.config.provider, model=self._harness.config.model,
                cwd=self._config.cwd, messages=self._harness.messages, content=content, index=index,
            )
            self._harness.config.system += format_memories_for_prompt(memories)
        except Exception as exc:
            logger.warning("Memory retrieval failed: %s", exc)

    async def _try_extract_memories(
        self, messages: Sequence[AgentMessage],
    ) -> None:
        """提取候选并逐条保存；失败不影响已经完成的聊天。"""
        try:
            dialogue = build_memory_dialogue(messages)
            if not dialogue:
                return
            memory_index = await asyncio.to_thread(load_memory_index, self._config.cwd) # 把同步函数放到线程里执行
            prompt = build_memory_extraction_prompt(dialogue, memory_index)
            candidates = await extract_memory_candidates(
                self._harness.config.provider, model=self._harness.config.model, prompt=prompt,
            )
            if not candidates:
                return
            existing = await asyncio.to_thread(load_existing_memories, self._config.cwd, memory_index)
            stored_any = False
            for candidate in candidates:
                if not should_store_memory(candidate, existing):
                    continue
                try:
                    await asyncio.to_thread(store_memory, self._config.cwd, candidate)
                except Exception as exc:
                    logger.warning("Memory storage failed: %s", exc)
                    continue
                # 正文和索引都写入成功后，下一条候选才能看到这条记忆。
                existing.append(candidate)
                stored_any = True
            if stored_any:
                await self._try_consolidate_memories()
        except Exception as exc:
            logger.warning("Memory extraction failed: %s", exc)

    async def _try_consolidate_memories(self) -> None:
        """新增记忆后检查阈值；整合失败保留现有记忆，不影响聊天。"""
        threshold = self._config.memory_consolidation_threshold
        if threshold <= 0:
            return
        try:
            snapshot = await asyncio.to_thread(load_memory_snapshot, self._config.cwd)
            if len(snapshot.memories) <= threshold:
                return
            candidates = await consolidate_memory_candidates(
                self._harness.config.provider, model=self._harness.config.model,
                memories=snapshot.memories, threshold=threshold,
            )
            await asyncio.to_thread(rebuild_memories, self._config.cwd, candidates, snapshot)
        except Exception as exc:
            logger.warning("Memory consolidation failed: %s", exc)

    def _persist_new_messages(self, before_count: int) -> None: 
        """同步追加本次新增消息，再从磁盘记录更新会话状态。"""
        for message in self._harness.messages[before_count:]:
            self._config.storage.append(MessageEntry(message=message))
        self._refresh_persisted_state() # tau也是先存磁盘再更新状态


def is_context_overflow_error(event: ErrorEvent) -> bool:
    """只识别 provider 的明确上下文错误，包含 HTTP 错误响应 body。"""
    if event.source != "provider":
        return False
    text = event.message
    if event.data is not None:
        body = event.data.get("body")
        if isinstance(body, str):
            text += "\n" + body
    normalized = text.lower()
    if "rate limit" in normalized or "rate_limit" in normalized:
        return False
    return any(marker in normalized for marker in (
        "context_length_exceeded", "context_window_exceeded", "context length",
        "context window", "context limit", "maximum context", "max context",
        "input is too long", "prompt is too long", "too many tokens",
    ))
