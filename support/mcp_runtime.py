import asyncio
import inspect
import json
import re
import shlex
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


def _safe_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        value = "server"
    if value[0].isdigit():
        value = f"s_{value}"
    return value


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return shlex.split(value)
    return [str(value)]


def _as_mapping(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(val) for key, val in value.items()}


def _tool_attr(tool: Any, snake_name: str, camel_name: str, default: Any = None) -> Any:
    if isinstance(tool, dict):
        return tool.get(snake_name, tool.get(camel_name, default))
    return getattr(tool, snake_name, getattr(tool, camel_name, default))


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=_json_default)
    except Exception:
        return str(value)


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


@dataclass
class MCPServerConfig:
    name: str
    transport: str
    command: str = ""
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class MCPRuntimeConfig:
    servers: List[MCPServerConfig] = field(default_factory=list)
    auto_execute: bool = True
    max_agent_steps: int = 6
    tool_timeout_sec: float = 60.0
    max_tool_result_chars: int = 12000

    @property
    def enabled(self) -> bool:
        return bool(self.servers)


@dataclass
class MCPToolDefinition:
    namespaced_name: str
    server_name: str
    original_name: str
    description: str
    input_schema: Dict[str, Any]

    def as_openai_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.namespaced_name,
                "description": self.description,
                "parameters": self.input_schema or {"type": "object", "properties": {}, "required": []},
            },
        }


@dataclass
class MCPCallResult:
    content: str
    is_error: bool = False
    structured_content: Any = None


def _normalize_servers(raw_config: Any) -> List[MCPServerConfig]:
    if not raw_config:
        return []

    if isinstance(raw_config, str):
        raw_config = json.loads(raw_config)

    if not isinstance(raw_config, dict):
        raise ValueError("MCP config must be a JSON object.")

    server_block = (
        raw_config.get("servers")
        or raw_config.get("mcpServers")
        or raw_config.get("mcp_servers")
        or raw_config.get("mcp")
        or {}
    )

    raw_servers = []
    if isinstance(server_block, dict):
        for name, value in server_block.items():
            if isinstance(value, dict):
                item = value.copy()
                item.setdefault("name", name)
                raw_servers.append(item)
    elif isinstance(server_block, list):
        raw_servers = [item for item in server_block if isinstance(item, dict)]
    else:
        raise ValueError("MCP config servers must be an object or list.")

    servers = []
    used_names = set()
    for index, item in enumerate(raw_servers, start=1):
        if item.get("disabled") is True:
            continue

        display_name = str(item.get("name") or f"server_{index}")
        safe = _safe_name(display_name)
        if safe in used_names:
            suffix = 2
            while f"{safe}_{suffix}" in used_names:
                suffix += 1
            safe = f"{safe}_{suffix}"
        used_names.add(safe)

        transport = str(item.get("transport") or "").lower().strip()
        url = str(item.get("url") or item.get("endpoint") or "").strip()
        command = str(item.get("command") or "").strip()

        if not transport:
            transport = "http" if url else "stdio"
        if transport in {"streamable_http", "streamable-http", "http"}:
            transport = "http"
        elif transport != "stdio":
            raise ValueError(f'Unsupported MCP transport "{transport}" for server "{display_name}".')

        if transport == "http" and not url:
            raise ValueError(f'MCP HTTP server "{display_name}" requires a url.')
        if transport == "stdio" and not command:
            raise ValueError(f'MCP stdio server "{display_name}" requires a command.')

        servers.append(
            MCPServerConfig(
                name=safe,
                transport=transport,
                command=command,
                args=_as_list(item.get("args")),
                env=_as_mapping(item.get("env")),
                cwd=str(item.get("cwd")) if item.get("cwd") else None,
                url=url,
                headers=_as_mapping(item.get("headers")),
            )
        )

    return servers


def parse_mcp_config(
    mcp_json: Any,
    max_agent_steps: int = 6,
    tool_timeout_sec: float = 60.0,
    max_tool_result_chars: int = 12000,
    auto_execute: bool = True,
) -> MCPRuntimeConfig:
    raw_config: Dict[str, Any] = {}
    if isinstance(mcp_json, str) and mcp_json.strip():
        raw_config = json.loads(mcp_json)
    elif isinstance(mcp_json, dict):
        raw_config = mcp_json
    elif mcp_json in (None, ""):
        raw_config = {}
    else:
        raise ValueError("MCP config must be a JSON object or empty string.")

    return MCPRuntimeConfig(
        servers=_normalize_servers(raw_config),
        auto_execute=bool(raw_config.get("auto_execute", auto_execute)),
        max_agent_steps=max(1, int(raw_config.get("max_agent_steps", max_agent_steps))),
        tool_timeout_sec=max(0.1, float(raw_config.get("tool_timeout_sec", tool_timeout_sec))),
        max_tool_result_chars=max(1, int(raw_config.get("max_tool_result_chars", max_tool_result_chars))),
    )


def format_mcp_result(result: Any, max_chars: int) -> MCPCallResult:
    is_error = bool(getattr(result, "is_error", getattr(result, "isError", False)))
    structured = getattr(result, "structured_content", getattr(result, "structuredContent", None))
    parts: List[str] = []

    if structured is not None:
        parts.append(_json_dumps(structured))

    content = getattr(result, "content", None)
    if content:
        for block in content:
            block_type = _tool_attr(block, "type", "type", "")
            text = _tool_attr(block, "text", "text", None)
            if text is not None:
                parts.append(str(text))
            elif block_type:
                parts.append(f"[{block_type} content omitted]")
            else:
                parts.append(_json_dumps(block))

    if not parts:
        parts.append(_json_dumps(result))

    text = "\n".join(part for part in parts if part)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + f"\n\n[truncated to {max_chars} characters]"

    return MCPCallResult(content=text, is_error=is_error, structured_content=structured)


class MCPToolbox:
    def __init__(
        self,
        config: MCPRuntimeConfig,
        client_factory: Optional[Callable[[MCPServerConfig], Any]] = None,
    ):
        self.config = config
        self.client_factory = client_factory
        self._stack: Optional[AsyncExitStack] = None
        self._clients: Dict[str, Any] = {}
        self._tool_map: Dict[str, MCPToolDefinition] = {}

    async def __aenter__(self) -> "MCPToolbox":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        try:
            for server in self.config.servers:
                client_cm = self.client_factory(server) if self.client_factory else self._default_client_context(server)
                client = await asyncio.wait_for(
                    self._stack.enter_async_context(client_cm),
                    timeout=self.config.tool_timeout_sec,
                )
                self._clients[server.name] = client
                await self._load_server_tools(server, client)
        except Exception:
            await self.__aexit__(None, None, None)
            raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(exc_type, exc_val, exc_tb)
        self._stack = None
        self._clients.clear()
        self._tool_map.clear()

    def _default_client_context(self, server: MCPServerConfig) -> Any:
        try:
            from mcp import Client, StdioServerParameters
        except Exception as exc:
            raise RuntimeError('MCP support requires installing "mcp>=2,<3".') from exc

        if server.transport == "stdio":
            from mcp.client.stdio import stdio_client

            kwargs: Dict[str, Any] = {
                "command": server.command,
                "args": server.args,
                "env": server.env or None,
            }
            if server.cwd and "cwd" in inspect.signature(StdioServerParameters).parameters:
                kwargs["cwd"] = server.cwd
            params = StdioServerParameters(**kwargs)
            return Client(stdio_client(params))

        if server.headers:
            try:
                import httpx2  # type: ignore
                from mcp.client.streamable_http import streamable_http_client
            except Exception as exc:
                raise RuntimeError('MCP HTTP headers require the MCP SDK HTTP dependencies.') from exc

            http_client = httpx2.AsyncClient(
                headers=server.headers,
                timeout=httpx2.Timeout(30.0, read=max(30.0, self.config.tool_timeout_sec)),
                follow_redirects=True,
            )

            class _HTTPClientContext:
                async def __aenter__(inner_self):
                    stack = AsyncExitStack()
                    await stack.__aenter__()
                    await stack.enter_async_context(http_client)
                    transport = streamable_http_client(server.url, http_client=http_client)
                    client = await stack.enter_async_context(Client(transport))
                    inner_self._stack = stack
                    return client

                async def __aexit__(inner_self, exc_type, exc_val, exc_tb):
                    await inner_self._stack.__aexit__(exc_type, exc_val, exc_tb)

            return _HTTPClientContext()

        return Client(server.url)

    async def _load_server_tools(self, server: MCPServerConfig, client: Any) -> None:
        result = await asyncio.wait_for(client.list_tools(), timeout=self.config.tool_timeout_sec)
        tools = getattr(result, "tools", [])
        for tool in tools:
            original_name = str(_tool_attr(tool, "name", "name", ""))
            if not original_name:
                continue

            namespaced_name = f"{server.name}__{_safe_name(original_name)}"
            if namespaced_name in self._tool_map:
                suffix = 2
                while f"{namespaced_name}_{suffix}" in self._tool_map:
                    suffix += 1
                namespaced_name = f"{namespaced_name}_{suffix}"

            title = _tool_attr(tool, "title", "title", "")
            description = _tool_attr(tool, "description", "description", "") or title or original_name
            input_schema = _tool_attr(tool, "input_schema", "inputSchema", None) or {}
            if hasattr(input_schema, "model_dump"):
                input_schema = input_schema.model_dump()
            if not isinstance(input_schema, dict):
                input_schema = {"type": "object", "properties": {}, "required": []}

            self._tool_map[namespaced_name] = MCPToolDefinition(
                namespaced_name=namespaced_name,
                server_name=server.name,
                original_name=original_name,
                description=str(description),
                input_schema=input_schema,
            )

    def openai_tools(self) -> List[Dict[str, Any]]:
        return [tool.as_openai_tool() for tool in self._tool_map.values()]

    def tool_names(self) -> List[str]:
        return list(self._tool_map.keys())

    async def call_tool(self, namespaced_name: str, arguments: Dict[str, Any]) -> MCPCallResult:
        if not self.config.auto_execute:
            return MCPCallResult(
                content=f'Tool "{namespaced_name}" was not executed because auto_execute is disabled.',
                is_error=True,
            )

        definition = self._tool_map.get(namespaced_name)
        if definition is None:
            return MCPCallResult(content=f'Unknown MCP tool "{namespaced_name}".', is_error=True)

        client = self._clients.get(definition.server_name)
        if client is None:
            return MCPCallResult(content=f'MCP server "{definition.server_name}" is not connected.', is_error=True)

        try:
            result = await asyncio.wait_for(
                client.call_tool(definition.original_name, arguments or {}),
                timeout=self.config.tool_timeout_sec,
            )
            return format_mcp_result(result, self.config.max_tool_result_chars)
        except asyncio.TimeoutError:
            return MCPCallResult(
                content=f'Tool "{namespaced_name}" timed out after {self.config.tool_timeout_sec:.1f}s.',
                is_error=True,
            )
        except Exception as exc:
            return MCPCallResult(content=f'Tool "{namespaced_name}" failed: {exc}', is_error=True)
