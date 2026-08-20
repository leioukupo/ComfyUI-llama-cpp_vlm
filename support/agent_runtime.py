import asyncio
import hashlib
import inspect
import json
import re
import threading
from dataclasses import dataclass, field
from html import unescape
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from .mcp_runtime import MCPCallResult, MCPRuntimeConfig, MCPToolbox
from .skill_runtime import SkillLibrary, skill_tool_specs


AGENT_SYSTEM_PROMPT = """You are running inside ComfyUI with optional local skills and MCP tools.
Local skills are private workflow instructions. Never recommend them, summarize them, or list them to the user unless the user explicitly asks for a skill catalog.
Use tools only when they materially help with the user's request. If a local skill is relevant, call skill_list first, then skill_read before following detailed skill instructions.
Tool calls are executed automatically by the workflow. Do not claim that a tool was used unless the tool result is present in the conversation.
If native tool calling is unavailable, request tools by replying with only this JSON shape:
{"tool_calls":[{"name":"tool_name","arguments":{}}]}
After tool results are returned, continue normally and provide the final answer only."""


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?(?:</think>|$)", "", text or "", flags=re.DOTALL).strip()


def run_coro_sync(coro: Awaitable[Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: Dict[str, Any] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 - relay to caller thread.
            result["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "error" in result:
        raise result["error"]
    return result.get("value")


class _AsyncLoopThread:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        self.loop.close()

    def call(self, coro: Awaitable[Any]) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()


class SyncMCPToolbox:
    def __init__(self, config: MCPRuntimeConfig):
        self.config = config
        self.loop_thread: Optional[_AsyncLoopThread] = None
        self.toolbox: Optional[MCPToolbox] = None

    def __enter__(self) -> "SyncMCPToolbox":
        self.loop_thread = _AsyncLoopThread()
        self.toolbox = MCPToolbox(self.config)
        try:
            self.loop_thread.call(self.toolbox.__aenter__())
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.loop_thread is not None and self.toolbox is not None:
            self.loop_thread.call(self.toolbox.__aexit__(exc_type, exc_val, exc_tb))
        if self.loop_thread is not None:
            self.loop_thread.close()
        self.loop_thread = None
        self.toolbox = None

    def openai_tools(self) -> List[Dict[str, Any]]:
        return self.toolbox.openai_tools() if self.toolbox is not None else []

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> MCPCallResult:
        if self.loop_thread is None or self.toolbox is None:
            return MCPCallResult(content="MCP toolbox is not connected.", is_error=True)
        return self.loop_thread.call(self.toolbox.call_tool(name, arguments))


@dataclass
class AgentToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    source: str = "fallback"


@dataclass
class AgentRunResult:
    output: str
    trace: List[Dict[str, Any]]
    selected_skills: List[str]
    transcript: List[Dict[str, Any]] = field(default_factory=list)

    def trace_json(self) -> str:
        return json.dumps(self.trace, ensure_ascii=False, indent=2)

    def selected_skills_json(self) -> str:
        return json.dumps(self.selected_skills, ensure_ascii=False)

    def transcript_json(self) -> str:
        return json.dumps(self.transcript, ensure_ascii=False, indent=2)


@dataclass
class ConversationSession:
    state_uid: str
    signature: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0
    last_output: str = ""

    def compatible(self, signature: str) -> bool:
        return bool(self.signature) and self.signature == signature

    def clone(self) -> "ConversationSession":
        return ConversationSession.from_dict(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_uid": self.state_uid,
            "signature": self.signature,
            "messages": _clone_messages(self.messages),
            "turn_count": self.turn_count,
            "last_output": self.last_output,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ConversationSession":
        if isinstance(data, ConversationSession):
            return data.clone()
        if not isinstance(data, dict):
            raise TypeError("ConversationSession data must be a dict or ConversationSession.")
        return cls(
            state_uid=str(data.get("state_uid") or ""),
            signature=str(data.get("signature") or ""),
            messages=_clone_messages(data.get("messages") or []),
            turn_count=int(data.get("turn_count") or 0),
            last_output=str(data.get("last_output") or ""),
        )


def normalize_session(session: Any) -> Optional[ConversationSession]:
    if session in (None, ""):
        return None
    if isinstance(session, ConversationSession):
        return session.clone()
    if isinstance(session, dict):
        return ConversationSession.from_dict(session)
    if hasattr(session, "to_dict"):
        return ConversationSession.from_dict(session.to_dict())
    if hasattr(session, "__dict__"):
        return ConversationSession.from_dict(session.__dict__)
    return None


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default, sort_keys=True, separators=(",", ":"))


def build_session_signature(payload: Dict[str, Any]) -> str:
    blob = _stable_json_dumps(payload)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _parse_arguments(arguments: Any) -> Dict[str, Any]:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _message_from_completion(completion: Dict[str, Any]) -> Dict[str, Any]:
    choices = completion.get("choices") or []
    if not choices:
        return {"role": "assistant", "content": ""}

    choice = choices[0]
    message = choice.get("message")
    if isinstance(message, dict):
        return message

    return {"role": "assistant", "content": choice.get("text", "")}


def _strip_code_fence(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return value


def _load_jsonish(text: str) -> Optional[Any]:
    value = _strip_code_fence(strip_thinking(text))
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        pass

    start = value.find("{")
    end = value.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(value[start : end + 1])
    except Exception:
        return None


def _normalize_fallback_calls(data: Any, tool_names: Iterable[str]) -> List[AgentToolCall]:
    known = set(tool_names)
    raw_calls = []

    if isinstance(data, dict):
        if isinstance(data.get("tool_calls"), list):
            raw_calls = data["tool_calls"]
        elif isinstance(data.get("tool_call"), dict):
            raw_calls = [data["tool_call"]]
        elif data.get("tool") or data.get("name"):
            raw_calls = [data]
    elif isinstance(data, list):
        raw_calls = data

    calls: List[AgentToolCall] = []
    for index, item in enumerate(raw_calls, start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("tool") or "").strip()
        if name not in known:
            continue
        arguments = _parse_arguments(item.get("arguments", item.get("args", {})))
        calls.append(AgentToolCall(name=name, arguments=arguments, call_id=f"fallback_{index}", source="fallback"))
    return calls


def _parse_xml_parameter_value(value: str) -> Any:
    value = unescape((value or "").strip())
    if not value:
        return ""
    try:
        return json.loads(value)
    except Exception:
        return value


def _extract_xml_tool_calls(text: str, tool_names: Iterable[str]) -> List[AgentToolCall]:
    known = set(tool_names)
    value = strip_thinking(text or "")
    calls: List[AgentToolCall] = []
    blocks = re.findall(r"<tool_call\b[^>]*>(.*?)</tool_call>", value, flags=re.DOTALL | re.IGNORECASE)

    for index, block in enumerate(blocks, start=1):
        fn_match = re.search(
            r"<function\s*=\s*([A-Za-z0-9_.:-]+)\s*>(.*?)</function>",
            block,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fn_match:
            name = fn_match.group(1).strip()
            body = fn_match.group(2)
        else:
            name_match = re.search(r"<name\b[^>]*>(.*?)</name>", block, flags=re.DOTALL | re.IGNORECASE)
            name = unescape(name_match.group(1).strip()) if name_match else ""
            body = block

        if name not in known:
            continue

        arguments: Dict[str, Any] = {}
        args_match = re.search(r"<arguments\b[^>]*>(.*?)</arguments>", body, flags=re.DOTALL | re.IGNORECASE)
        if args_match:
            arguments = _parse_arguments(unescape(args_match.group(1).strip()))

        for param_name, param_value in re.findall(
            r"<parameter\s*=\s*([A-Za-z0-9_.:-]+)\s*>(.*?)</parameter>",
            body,
            flags=re.DOTALL | re.IGNORECASE,
        ):
            arguments[param_name.strip()] = _parse_xml_parameter_value(param_value)

        calls.append(AgentToolCall(name=name, arguments=arguments, call_id=f"xml_{index}", source="fallback"))

    return calls


def extract_tool_calls(message: Dict[str, Any], tool_names: Iterable[str]) -> List[AgentToolCall]:
    calls: List[AgentToolCall] = []

    for index, item in enumerate(message.get("tool_calls") or [], start=1):
        if not isinstance(item, dict):
            continue
        fn = item.get("function") or {}
        name = str(fn.get("name") or item.get("name") or "").strip()
        if not name:
            continue
        calls.append(
            AgentToolCall(
                name=name,
                arguments=_parse_arguments(fn.get("arguments", item.get("arguments"))),
                call_id=str(item.get("id") or f"call_{index}"),
                source="native",
            )
        )

    function_call = message.get("function_call")
    if isinstance(function_call, dict):
        name = str(function_call.get("name") or "").strip()
        if name:
            calls.append(
                AgentToolCall(
                    name=name,
                    arguments=_parse_arguments(function_call.get("arguments")),
                    call_id="function_call",
                    source="native",
                )
            )

    if calls:
        return calls

    content = str(message.get("content") or "")
    data = _load_jsonish(content)
    calls = _normalize_fallback_calls(data, tool_names)
    if calls:
        return calls
    return _extract_xml_tool_calls(content, tool_names)


def _tool_name(tool: Dict[str, Any]) -> str:
    return str(tool.get("function", {}).get("name") or "")


def _format_tools_for_prompt(tools: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    compact = []
    for tool in tools:
        fn = tool.get("function", {})
        compact.append(
            {
                "name": fn.get("name"),
                "description": fn.get("description"),
                "parameters": fn.get("parameters"),
            }
        )
    text = _json_dumps(compact)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n[tool schemas truncated]"
    return text


def _clone_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return json.loads(json.dumps(messages))


def _llm_context_size(llm: Any) -> int:
    for attr in ("n_ctx", "_n_ctx"):
        value = getattr(llm, attr, None)
        try:
            value = value() if callable(value) else value
            if value:
                return int(value)
        except Exception:
            continue
    return 8192


def _default_skill_read_chars(n_ctx: int) -> int:
    if n_ctx <= 8192:
        return 4000
    if n_ctx <= 16384:
        return 8000
    if n_ctx <= 32768:
        return 16000
    return 24000


def _is_context_overflow_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "context shift" in text or ("n_ctx" in text and "context" in text)


def _context_overflow_message(n_ctx: int) -> str:
    return (
        "Agent context exceeded the llama.cpp context window. "
        f"Current n_ctx is {n_ctx}. Increase Llama-cpp Model Loader n_ctx "
        "(32768 or 65536 is recommended for Qwen3.5 Agent workflows), "
        "clear saved conversation state, or reduce Skill Library max_skill_chars / MCP max_tool_result_chars."
    )


def _sanitize_message_for_transcript(message: Dict[str, Any]) -> Dict[str, Any]:
    cloned = json.loads(json.dumps(message))
    content = cloned.get("content")
    if isinstance(content, str):
        cloned["content"] = strip_thinking(content)
    elif isinstance(content, list):
        sanitized_content = []
        for item in content:
            if isinstance(item, dict):
                cloned_item = json.loads(json.dumps(item))
                if cloned_item.get("type") == "text":
                    cloned_item["text"] = strip_thinking(str(cloned_item.get("text") or ""))
                sanitized_content.append(cloned_item)
            else:
                sanitized_content.append(item)
        cloned["content"] = sanitized_content
    return cloned


class AgentRunner:
    def __init__(
        self,
        llm: Any,
        seed: int,
        parameters: Optional[Dict[str, Any]] = None,
        skill_library: Optional[SkillLibrary] = None,
        mcp_config: Optional[MCPRuntimeConfig] = None,
    ):
        self.llm = llm
        self.seed = seed
        self.parameters = (parameters or {}).copy()
        self.skill_library = skill_library
        self.mcp_config = mcp_config or MCPRuntimeConfig()
        self.native_tools_supported = True

    def run(self, messages: List[Dict[str, Any]]) -> AgentRunResult:
        return self._run_sync(messages)

    async def arun(self, messages: List[Dict[str, Any]]) -> AgentRunResult:
        return self._run_sync(messages)

    def _run_sync(self, messages: List[Dict[str, Any]]) -> AgentRunResult:
        trace: List[Dict[str, Any]] = []
        tool_callbacks: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        tool_metadata: Dict[str, Dict[str, str]] = {}
        tools: List[Dict[str, Any]] = []
        selected_skills: List[str] = []

        if self.skill_library is not None:
            skill_language = self.skill_library.effective_language(_messages_text(messages))
            for spec in skill_tool_specs():
                name = _tool_name(spec)
                if name:
                    tools.append(spec)
                    tool_callbacks[name] = lambda args, tool_name=name, lang=skill_language: self._call_skill_tool(tool_name, args, lang)
                    tool_metadata[name] = {"kind": "skill", "provider": "local"}

        with SyncMCPToolbox(self.mcp_config) as mcp_tools:
            for spec in mcp_tools.openai_tools():
                name = _tool_name(spec)
                if name:
                    tools.append(spec)
                    tool_callbacks[name] = lambda args, tool_name=name: mcp_tools.call_tool(tool_name, args)
                    tool_metadata[name] = {"kind": "mcp", "provider": name.split("__", 1)[0]}

            working_messages = self._prepare_messages(messages, tools)
            transcript = [message for message in _clone_messages(messages) if message.get("role") != "system"]
            tool_names = [_tool_name(tool) for tool in tools]
            max_steps = max(1, self.mcp_config.max_agent_steps)

            for step in range(1, max_steps + 1):
                try:
                    completion = self._create_chat_completion(working_messages, tools)
                except RuntimeError as exc:
                    if not _is_context_overflow_error(exc):
                        raise
                    message = _context_overflow_message(_llm_context_size(self.llm))
                    trace.append({"step": step, "error": message, "exception": str(exc)[:4000], "branch": "error"})
                    return AgentRunResult(output=message, trace=trace, selected_skills=selected_skills, transcript=transcript)
                assistant_message = _message_from_completion(completion)
                content = str(assistant_message.get("content") or "")
                calls = extract_tool_calls(assistant_message, tool_names)
                branch = "native" if calls and calls[0].source == "native" else "fallback" if calls else "final"

                trace.append(
                    {
                        "step": step,
                        "assistant": strip_thinking(content)[:4000],
                        "branch": branch,
                        "tool_calls": [
                            {"name": call.name, "arguments": call.arguments, "source": call.source}
                            for call in calls
                        ],
                    }
                )

                if not calls:
                    sanitized_assistant = _sanitize_message_for_transcript(assistant_message)
                    if not sanitized_assistant.get("content") and strip_thinking(content):
                        sanitized_assistant["content"] = strip_thinking(content)
                    transcript.append(sanitized_assistant)
                    return AgentRunResult(
                        output=strip_thinking(content),
                        trace=trace,
                        selected_skills=selected_skills,
                        transcript=transcript,
                    )

                if calls[0].source == "native":
                    sanitized_assistant = _sanitize_message_for_transcript(assistant_message)
                    working_messages.append(sanitized_assistant)
                    transcript.append(sanitized_assistant)
                    for call in calls:
                        result = self._execute_tool(call, tool_callbacks, selected_skills)
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": call.call_id,
                            "name": call.name,
                            "content": result.content,
                        }
                        working_messages.append(tool_message)
                        transcript.append(tool_message)
                        trace.append(self._trace_tool_result(step, call, result, tool_metadata.get(call.name, {})))
                else:
                    fallback_call_summaries = [
                        {"name": call.name, "arguments": call.arguments}
                        for call in calls
                    ]
                    synthetic_assistant = {
                        "role": "assistant",
                        "content": "Requested tool calls:\n" + _json_dumps(fallback_call_summaries),
                    }
                    working_messages.append(synthetic_assistant)
                    transcript.append(synthetic_assistant)
                    fallback_results = []
                    for call in calls:
                        result = self._execute_tool(call, tool_callbacks, selected_skills)
                        fallback_results.append(
                            {
                                "name": call.name,
                                "arguments": call.arguments,
                                "is_error": result.is_error,
                                "content": result.content,
                            }
                        )
                        trace.append(self._trace_tool_result(step, call, result, tool_metadata.get(call.name, {})))
                    tool_result_message = {
                        "role": "user",
                        "content": "Tool results:\n"
                        + _json_dumps(fallback_results)
                        + "\nUse these results to continue or provide the final answer.",
                    }
                    working_messages.append(tool_result_message)
                    transcript.append(tool_result_message)

            return AgentRunResult(
                output="Agent stopped because max_agent_steps was reached.",
                trace=trace,
                selected_skills=selected_skills,
                transcript=transcript,
            )

    def _prepare_messages(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cloned = _clone_messages(messages)
        hint_text = _messages_text(cloned)
        prompt_parts = [AGENT_SYSTEM_PROMPT]
        if self.skill_library is not None:
            prompt_parts.append(self.skill_library.as_system_context(hint_text))
        if tools:
            prompt_parts.append("Available tool schemas:\n" + _format_tools_for_prompt(tools))
        return [{"role": "system", "content": "\n\n".join(prompt_parts)}] + cloned

    def _create_chat_completion(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        kwargs = self.parameters.copy()
        if tools and self.native_tools_supported:
            try:
                return self.llm.create_chat_completion(
                    messages=messages,
                    seed=self.seed,
                    tools=tools,
                    tool_choice="auto",
                    **kwargs,
                )
            except TypeError as exc:
                if "tool" not in str(exc):
                    raise
                self.native_tools_supported = False

        return self.llm.create_chat_completion(messages=messages, seed=self.seed, **kwargs)

    def _execute_tool(
        self,
        call: AgentToolCall,
        callbacks: Dict[str, Callable[[Dict[str, Any]], Any]],
        selected_skills: List[str],
    ) -> MCPCallResult:
        callback = callbacks.get(call.name)
        if callback is None:
            return MCPCallResult(content=f'Unknown tool "{call.name}".', is_error=True)

        try:
            result = callback(call.arguments)
            if inspect.isawaitable(result):
                result = run_coro_sync(result)
            if isinstance(result, MCPCallResult):
                self._record_skill_use(call, result, selected_skills)
                return result
            text = str(result)
            limit = self.mcp_config.max_tool_result_chars
            if len(text) > limit:
                text = text[:limit].rstrip() + f"\n\n[truncated to {limit} characters]"
            wrapped = MCPCallResult(content=text, is_error=False)
            self._record_skill_use(call, wrapped, selected_skills)
            return wrapped
        except Exception as exc:
            return MCPCallResult(content=f'Tool "{call.name}" failed: {exc}', is_error=True)

    def _record_skill_use(self, call: AgentToolCall, result: MCPCallResult, selected_skills: List[str]) -> None:
        if self.skill_library is None or call.name != "skill_read" or result.is_error:
            return

        skill_name = str(call.arguments.get("skill_name") or call.arguments.get("name") or "").strip()
        if not skill_name:
            return

        try:
            canonical_name = self.skill_library.canonical_name(skill_name)
        except Exception:
            canonical_name = skill_name

        if canonical_name and canonical_name not in selected_skills:
            selected_skills.append(canonical_name)

    def _call_skill_tool(self, tool_name: str, arguments: Dict[str, Any], language: str) -> str:
        args = (arguments or {}).copy()
        if args.get("language") in (None, "", "auto"):
            args["language"] = language
        if tool_name == "skill_read" and not args.get("max_chars"):
            args["max_chars"] = min(
                self.skill_library.max_skill_chars,
                self.mcp_config.max_tool_result_chars,
                _default_skill_read_chars(_llm_context_size(self.llm)),
            )
        return self.skill_library.call_tool(tool_name, args)

    def _trace_tool_result(
        self,
        step: int,
        call: AgentToolCall,
        result: MCPCallResult,
        metadata: Dict[str, str],
    ) -> Dict[str, Any]:
        return {
            "step": step,
            "tool": call.name,
            "tool_kind": metadata.get("kind", "unknown"),
            "tool_provider": metadata.get("provider", ""),
            "source": call.source,
            "arguments": call.arguments,
            "is_error": result.is_error,
            "content": result.content[:4000],
        }


def _messages_text(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
    return "\n".join(parts)
