import asyncio
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from .mcp_runtime import MCPCallResult, MCPRuntimeConfig, MCPToolbox
from .skill_runtime import SkillLibrary, skill_tool_specs


AGENT_SYSTEM_PROMPT = """You are running inside ComfyUI with optional local skills and MCP tools.
Use tools only when they materially help with the user's request. If local skills are relevant, call skill_read before following detailed skill instructions.
Tool calls are executed automatically by the workflow. Do not claim that a tool was used unless the tool result is present in the conversation.
If native tool calling is unavailable, request tools by replying with only this JSON shape:
{"tool_calls":[{"name":"tool_name","arguments":{}}]}
After tool results are returned, continue normally and provide the final answer."""


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

    def trace_json(self) -> str:
        return json.dumps(self.trace, ensure_ascii=False, indent=2)

    def selected_skills_json(self) -> str:
        return json.dumps(self.selected_skills, ensure_ascii=False)


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

    data = _load_jsonish(str(message.get("content") or ""))
    return _normalize_fallback_calls(data, tool_names)


def _tool_name(tool: Dict[str, Any]) -> str:
    return str(tool.get("function", {}).get("name") or "")


def _format_tools_for_prompt(tools: List[Dict[str, Any]], max_chars: int = 8000) -> str:
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
        return run_coro_sync(self.arun(messages))

    async def arun(self, messages: List[Dict[str, Any]]) -> AgentRunResult:
        trace: List[Dict[str, Any]] = []
        tool_callbacks: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        tools: List[Dict[str, Any]] = []
        selected_skills: List[str] = []

        if self.skill_library is not None:
            skill_language = self.skill_library.effective_language(_messages_text(messages))
            selected_skills = self.skill_library.available_names()
            for spec in skill_tool_specs():
                name = _tool_name(spec)
                if name:
                    tools.append(spec)
                    tool_callbacks[name] = lambda args, tool_name=name, lang=skill_language: self._call_skill_tool(tool_name, args, lang)

        async with MCPToolbox(self.mcp_config) as mcp_tools:
            for spec in mcp_tools.openai_tools():
                name = _tool_name(spec)
                if name:
                    tools.append(spec)
                    tool_callbacks[name] = lambda args, tool_name=name: mcp_tools.call_tool(tool_name, args)

            working_messages = self._prepare_messages(messages, tools)
            tool_names = [_tool_name(tool) for tool in tools]
            max_steps = max(1, self.mcp_config.max_agent_steps)

            for step in range(1, max_steps + 1):
                completion = self._create_chat_completion(working_messages, tools)
                assistant_message = _message_from_completion(completion)
                content = str(assistant_message.get("content") or "")
                calls = extract_tool_calls(assistant_message, tool_names)

                trace.append(
                    {
                        "step": step,
                        "assistant": strip_thinking(content)[:4000],
                        "tool_calls": [
                            {"name": call.name, "arguments": call.arguments, "source": call.source}
                            for call in calls
                        ],
                    }
                )

                if not calls:
                    return AgentRunResult(output=strip_thinking(content), trace=trace, selected_skills=selected_skills)

                if calls[0].source == "native":
                    working_messages.append(assistant_message)
                    for call in calls:
                        result = await self._execute_tool(call, tool_callbacks)
                        working_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.call_id,
                                "name": call.name,
                                "content": result.content,
                            }
                        )
                        trace.append(self._trace_tool_result(step, call, result))
                else:
                    working_messages.append({"role": "assistant", "content": content})
                    fallback_results = []
                    for call in calls:
                        result = await self._execute_tool(call, tool_callbacks)
                        fallback_results.append(
                            {
                                "name": call.name,
                                "arguments": call.arguments,
                                "is_error": result.is_error,
                                "content": result.content,
                            }
                        )
                        trace.append(self._trace_tool_result(step, call, result))
                    working_messages.append(
                        {
                            "role": "user",
                            "content": "Tool results:\n"
                            + _json_dumps(fallback_results)
                            + "\nUse these results to continue or provide the final answer.",
                        }
                    )

            return AgentRunResult(
                output=strip_thinking(content) or "Agent stopped because max_agent_steps was reached.",
                trace=trace,
                selected_skills=selected_skills,
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

    async def _execute_tool(
        self,
        call: AgentToolCall,
        callbacks: Dict[str, Callable[[Dict[str, Any]], Any]],
    ) -> MCPCallResult:
        callback = callbacks.get(call.name)
        if callback is None:
            return MCPCallResult(content=f'Unknown tool "{call.name}".', is_error=True)

        try:
            result = callback(call.arguments)
            if asyncio.iscoroutine(result):
                result = await result
            if isinstance(result, MCPCallResult):
                return result
            text = str(result)
            limit = self.mcp_config.max_tool_result_chars
            if len(text) > limit:
                text = text[:limit].rstrip() + f"\n\n[truncated to {limit} characters]"
            return MCPCallResult(content=text, is_error=False)
        except Exception as exc:
            return MCPCallResult(content=f'Tool "{call.name}" failed: {exc}', is_error=True)

    def _call_skill_tool(self, tool_name: str, arguments: Dict[str, Any], language: str) -> str:
        args = (arguments or {}).copy()
        if args.get("language") in (None, "", "auto"):
            args["language"] = language
        return self.skill_library.call_tool(tool_name, args)

    def _trace_tool_result(self, step: int, call: AgentToolCall, result: MCPCallResult) -> Dict[str, Any]:
        return {
            "step": step,
            "tool": call.name,
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
