import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .agent_runtime import (
    AgentRunner,
    STATUS_ERROR as AGENT_STATUS_ERROR,
    _context_overflow_message,
    _is_context_overflow_error,
    _llm_context_size,
    _message_from_completion,
    _progress_placeholder_continue_prompt,
    build_session_signature,
    looks_like_progress_placeholder,
    patch_llama_template_compatibility,
    strip_thinking,
)


STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_WAITING = "waiting"
STATUS_STOPPING = "stopping"
STATUS_ENDED = "ended"
STATUS_ERROR = "error"


def _clone_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return json.loads(json.dumps(messages))


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return strip_thinking(content).strip()

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                elif item.get("text") is not None:
                    parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return strip_thinking("\n".join(parts)).strip()

    return strip_thinking(str(content or "")).strip()


def render_transcript(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        if role in {"system", "tool"}:
            continue
        text = _message_text(message)
        if not text:
            continue
        if text.startswith("Requested tool calls:") or text.startswith("Tool results:"):
            continue
        label = {
            "user": "User",
            "assistant": "Assistant",
        }.get(role, role.title() or "Message")
        lines.append(f"{label}: {text}")
    return "\n\n".join(lines).strip()


def build_live_session_signature(
    llama_model: Any,
    system_prompt: str,
    parameters: Optional[Dict[str, Any]] = None,
    skills: Any = None,
    mcp_config: Any = None,
) -> str:
    # Keep the selected model out of the signature so a live transcript can
    # continue across local model switches.
    return build_session_signature(
        {
            "mode": "live_chat",
            "system_prompt": system_prompt or "",
            "parameters": parameters or {},
            "skills": skills,
            "mcp_config": mcp_config,
        }
    )


def generate_chat_reply(llm: Any, seed: int, messages: List[Dict[str, Any]], parameters: Optional[Dict[str, Any]] = None) -> str:
    patch_llama_template_compatibility(llm)
    kwargs = (parameters or {}).copy()
    completion = llm.create_chat_completion(messages=messages, seed=seed, **kwargs)
    assistant_message = _message_from_completion(completion)
    text = str(assistant_message.get("content") or assistant_message.get("text") or "")
    return strip_thinking(text).strip()


@dataclass
class LiveChatSession:
    state_uid: str
    node_id: str
    signature: str
    system_prompt: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    pending_user_messages: List[str] = field(default_factory=list)
    turn_count: int = 0
    last_output: str = ""
    status: str = STATUS_IDLE
    stop_requested: bool = False
    error: str = ""
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    selected_skills: List[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    condition: threading.Condition = field(default_factory=threading.Condition, repr=False, compare=False)

    def touch(self) -> None:
        self.updated_at = time.time()

    def append_user_message(self, text: str) -> None:
        clean = strip_thinking(str(text or "")).strip()
        if not clean:
            return
        self.messages.append({"role": "user", "content": clean})
        self.touch()

    def append_assistant_message(self, text: str) -> None:
        clean = strip_thinking(str(text or "")).strip()
        self.messages.append({"role": "assistant", "content": clean})
        self.last_output = clean
        self.turn_count += 1
        self.touch()

    def queue_user_message(self, text: str) -> bool:
        clean = strip_thinking(str(text or "")).strip()
        if not clean or self.stop_requested or self.status in {STATUS_STOPPING, STATUS_ENDED, STATUS_ERROR}:
            return False
        with self.condition:
            self.pending_user_messages.append(clean)
            self.status = STATUS_RUNNING
            self.touch()
            self.condition.notify_all()
        return True

    def request_end(self) -> None:
        with self.condition:
            self.stop_requested = True
            self.status = STATUS_STOPPING
            self.touch()
            self.condition.notify_all()

    def snapshot(self) -> Dict[str, Any]:
        conversation = render_transcript(self.messages)
        if self.error:
            error_line = f"Error: {self.error}"
            conversation = f"{conversation}\n\n{error_line}".strip() if conversation else error_line
        return {
            "state_uid": self.state_uid,
            "node_id": self.node_id,
            "signature": self.signature,
            "system_prompt": self.system_prompt,
            "status": self.status,
            "turn_count": self.turn_count,
            "last_output": self.last_output,
            "conversation": conversation,
            "messages": _clone_messages(self.messages),
            "pending_count": len(self.pending_user_messages),
            "stop_requested": self.stop_requested,
            "error": self.error,
            "tool_trace": _clone_messages(self.tool_trace),
            "selected_skills": list(self.selected_skills),
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }


class LiveChatManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._sessions: Dict[str, LiveChatSession] = {}
        self._node_index: Dict[str, str] = {}

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
            self._node_index.clear()

    def _resolve_key(self, state_uid: Optional[str] = None, node_id: Optional[str] = None) -> Optional[str]:
        if state_uid not in (None, ""):
            return str(state_uid)
        if node_id not in (None, ""):
            with self._lock:
                return self._node_index.get(str(node_id))
        return None

    def get(self, state_uid: Optional[str] = None, node_id: Optional[str] = None) -> Optional[LiveChatSession]:
        key = self._resolve_key(state_uid, node_id)
        if not key:
            return None
        with self._lock:
            return self._sessions.get(key)

    def open_session(
        self,
        state_uid: str,
        node_id: str,
        signature: str,
        system_prompt: str,
        initial_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> LiveChatSession:
        key = str(state_uid or node_id)
        node_key = str(node_id or key)
        with self._lock:
            existing = self._sessions.get(key)
            if existing is None or existing.signature != signature or existing.status in {STATUS_ENDED, STATUS_ERROR}:
                existing = LiveChatSession(
                    state_uid=key,
                    node_id=node_key,
                    signature=signature,
                    system_prompt=system_prompt or "",
                    messages=_clone_messages(initial_messages or []),
                    status=STATUS_IDLE,
                )
                self._sessions[key] = existing
            else:
                existing.node_id = node_key
                existing.signature = signature
                existing.system_prompt = system_prompt or existing.system_prompt
                if initial_messages and not existing.messages:
                    existing.messages = _clone_messages(initial_messages)
            self._node_index[node_key] = key
            return existing

    def queue_message(self, state_uid: Optional[str] = None, node_id: Optional[str] = None, message: str = "") -> Optional[LiveChatSession]:
        session = self.get(state_uid=state_uid, node_id=node_id)
        if session is None:
            return None
        session.queue_user_message(message)
        return session

    def end_session(self, state_uid: Optional[str] = None, node_id: Optional[str] = None) -> Optional[LiveChatSession]:
        session = self.get(state_uid=state_uid, node_id=node_id)
        if session is None:
            return None
        session.request_end()
        return session

    def wait_for_message(
        self,
        session: LiveChatSession,
        should_stop: Optional[Callable[[], bool]] = None,
        poll_interval: float = 0.25,
    ) -> Optional[str]:
        with session.condition:
            while not session.pending_user_messages and not session.stop_requested:
                if should_stop is not None and should_stop():
                    session.stop_requested = True
                    session.status = STATUS_STOPPING
                    session.touch()
                    break
                session.condition.wait(timeout=poll_interval)
            if session.stop_requested or not session.pending_user_messages:
                return None
            message = session.pending_user_messages.pop(0)
            session.status = STATUS_RUNNING
            session.touch()
            return message

    def remove(self, state_uid: Optional[str] = None, node_id: Optional[str] = None) -> None:
        key = self._resolve_key(state_uid, node_id)
        if not key:
            return
        with self._lock:
            session = self._sessions.pop(key, None)
            if session is not None:
                self._node_index.pop(session.node_id, None)


def run_live_chat(
    manager: LiveChatManager,
    llm: Any,
    seed: int,
    parameters: Optional[Dict[str, Any]],
    session: LiveChatSession,
    should_stop: Callable[[], bool],
    emit_state: Callable[[LiveChatSession, str], None],
    initial_user_message: str = "",
    skill_library: Any = None,
    mcp_config: Any = None,
) -> LiveChatSession:
    full_messages = []
    if session.system_prompt.strip():
        full_messages.append({"role": "system", "content": session.system_prompt})
    if session.messages:
        full_messages.extend(_clone_messages(session.messages))
    use_agent = skill_library is not None or bool(getattr(mcp_config, "enabled", False))

    next_user_message = strip_thinking(str(initial_user_message or "")).strip()
    if initial_user_message.strip():
        session.status = STATUS_RUNNING
        emit_state(session, "started")

    if not session.messages and not next_user_message:
        session.status = STATUS_WAITING
        emit_state(session, "waiting")

    while True:
        if should_stop():
            session.stop_requested = True
            session.status = STATUS_STOPPING
            emit_state(session, "stopping")

        if session.stop_requested:
            break

        if not next_user_message:
            session.status = STATUS_WAITING
            emit_state(session, "waiting")
            next_user_message = manager.wait_for_message(session, should_stop=should_stop) or ""
            if not next_user_message:
                break

        session.append_user_message(next_user_message)
        full_messages.append({"role": "user", "content": next_user_message})
        next_user_message = ""
        session.status = STATUS_RUNNING
        emit_state(session, "user")

        try:
            if use_agent:
                runner = AgentRunner(
                    llm,
                    seed=seed,
                    parameters=parameters,
                    skill_library=skill_library,
                    mcp_config=mcp_config,
                )
                result = runner.run(full_messages)
                session.tool_trace.extend(result.trace)
                for name in result.selected_skills:
                    if name not in session.selected_skills:
                        session.selected_skills.append(name)
                if result.status == AGENT_STATUS_ERROR:
                    session.status = STATUS_ERROR
                    session.error = result.output or "Live chat agent failed."
                    emit_state(session, "error")
                    break

                session.messages = _clone_messages(result.transcript)
                session.last_output = strip_thinking(result.output)
                session.turn_count += 1
                full_messages = []
                if session.system_prompt.strip():
                    full_messages.append({"role": "system", "content": session.system_prompt})
                full_messages.extend(_clone_messages(session.messages))
                emit_state(session, "assistant")
                continue

            reply = generate_chat_reply(llm, seed, full_messages, parameters)
            for _ in range(2):
                if not looks_like_progress_placeholder(reply):
                    break
                session.tool_trace.append(
                    {
                        "step": len(session.tool_trace) + 1,
                        "branch": "progress_placeholder_continue",
                        "assistant": reply[:4000],
                    }
                )
                full_messages.append({"role": "assistant", "content": reply})
                full_messages.append({"role": "user", "content": _progress_placeholder_continue_prompt(reply)})
                reply = generate_chat_reply(llm, seed, full_messages, parameters)
        except Exception as exc:
            if isinstance(exc, RuntimeError) and _is_context_overflow_error(exc):
                reply = _context_overflow_message(_llm_context_size(llm))
                session.status = STATUS_ERROR
                session.error = reply
                emit_state(session, "error")
                break

            session.status = STATUS_ERROR
            session.error = f"Live chat generation failed: {exc}"
            emit_state(session, "error")
            break

        session.append_assistant_message(reply)
        full_messages.append({"role": "assistant", "content": reply})
        emit_state(session, "assistant")

    if session.status not in {STATUS_ERROR, STATUS_STOPPING}:
        session.status = STATUS_ENDED
    elif session.status == STATUS_STOPPING:
        session.status = STATUS_ENDED
    emit_state(session, "ended")
    return session
