import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from support.agent_runtime import AgentRunner, extract_tool_calls, strip_thinking
from support.mcp_runtime import MCPRuntimeConfig, MCPServerConfig, MCPToolbox, format_mcp_result, parse_mcp_config
from support.skill_runtime import scan_skill_directory


class TestSkillRuntime(unittest.TestCase):
    def test_scan_read_and_path_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "writer"
            refs = skill_dir / "references"
            refs.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: writer\ndescription: Write concise prompts.\n---\n# Writer\nEnglish body.",
                encoding="utf-8",
            )
            (skill_dir / "SKILL.cn.md").write_text("# 写作\n中文内容。", encoding="utf-8")
            (skill_dir / "meta.yaml").write_text(
                'summary-en: "English summary"\nsummary-cn: "中文摘要"\n',
                encoding="utf-8",
            )
            (refs / "guide.txt").write_text("reference text", encoding="utf-8")

            library = scan_skill_directory(str(root), selected_skills="writer", language="zh")

            self.assertEqual(library.available_names(), ["writer"])
            summaries = library.list_summaries("zh")
            self.assertEqual(summaries[0]["description"], "中文摘要")

            filtered = json.loads(library.call_tool("skill_list", {"query": "战斗 中文", "language": "zh"}))
            self.assertEqual(filtered["skills"][0]["name"], "writer")

            skill_payload = json.loads(library.read_skill("writer", language="zh"))
            self.assertEqual(skill_payload["path"], "SKILL.cn.md")
            self.assertIn("中文内容", skill_payload["content"])

            ref_payload = json.loads(library.read_skill("writer", path="references/guide.txt"))
            self.assertEqual(ref_payload["content"], "reference text")

            with self.assertRaises(ValueError):
                library.read_skill("writer", path="../secret.txt")


class TestMCPRuntime(unittest.TestCase):
    def test_parse_mcp_config_supports_stdio_and_http(self):
        config = parse_mcp_config(
            json.dumps(
                {
                    "mcpServers": {
                        "local tools": {
                            "command": "python3",
                            "args": ["server.py"],
                            "env": {"TOKEN": "abc"},
                        },
                        "remote": {
                            "transport": "streamable_http",
                            "url": "http://localhost:8000/mcp",
                            "headers": {"Authorization": "Bearer x"},
                        },
                    },
                    "max_agent_steps": 4,
                }
            ),
            tool_timeout_sec=2,
            max_tool_result_chars=128,
        )

        self.assertEqual(config.max_agent_steps, 4)
        self.assertEqual(config.servers[0].name, "local_tools")
        self.assertEqual(config.servers[0].transport, "stdio")
        self.assertEqual(config.servers[0].args, ["server.py"])
        self.assertEqual(config.servers[1].transport, "http")
        self.assertEqual(config.servers[1].url, "http://localhost:8000/mcp")

    def test_format_mcp_result_and_toolbox_mapping(self):
        class TextBlock:
            type = "text"
            text = "tool text"

        result = format_mcp_result(
            SimpleNamespace(content=[TextBlock()], structured_content={"value": 3}, is_error=False),
            max_chars=100,
        )
        self.assertFalse(result.is_error)
        self.assertIn('"value": 3', result.content)
        self.assertIn("tool text", result.content)

        class FakeClient:
            async def list_tools(self):
                return SimpleNamespace(
                    tools=[
                        SimpleNamespace(
                            name="add",
                            title="Add",
                            description="Add two numbers.",
                            input_schema={
                                "type": "object",
                                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                                "required": ["a", "b"],
                            },
                        )
                    ]
                )

            async def call_tool(self, name, arguments):
                self.last_call = (name, arguments)
                return SimpleNamespace(content=[TextBlock()], structured_content={"sum": 5}, is_error=False)

        class FakeContext:
            async def __aenter__(self):
                self.client = FakeClient()
                return self.client

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        async def run():
            cfg = MCPRuntimeConfig(
                servers=[MCPServerConfig(name="math", transport="stdio", command="fake")],
                tool_timeout_sec=1,
                max_tool_result_chars=1000,
            )
            async with MCPToolbox(cfg, client_factory=lambda _server: FakeContext()) as toolbox:
                tools = toolbox.openai_tools()
                self.assertEqual(tools[0]["function"]["name"], "math__add")
                call_result = await toolbox.call_tool("math__add", {"a": 2, "b": 3})
                self.assertFalse(call_result.is_error)
                self.assertIn('"sum": 5', call_result.content)

        asyncio.run(run())


class TestAgentRuntime(unittest.TestCase):
    def test_extract_fallback_tool_calls_and_strip_thinking(self):
        self.assertEqual(strip_thinking("hi <think>hidden</think> there"), "hi  there")
        calls = extract_tool_calls(
            {"content": '{"tool_calls":[{"name":"skill_list","arguments":{"query":"h3"}}]}'},
            ["skill_list"],
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "skill_list")
        self.assertEqual(calls[0].arguments["query"], "h3")

        xml_calls = extract_tool_calls(
            {
                "content": """为了查找技能：
<tool_call>
<function=skill_list>
<parameter=query>
战斗 动画 短片
</parameter>
<parameter=language>
zh
</parameter>
</function>
</tool_call>"""
            },
            ["skill_list"],
        )
        self.assertEqual(len(xml_calls), 1)
        self.assertEqual(xml_calls[0].name, "skill_list")
        self.assertEqual(xml_calls[0].arguments["query"], "战斗 动画 短片")
        self.assertEqual(xml_calls[0].arguments["language"], "zh")

    def test_agent_fallback_loop_executes_skill_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "demo"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: demo\ndescription: Demo skill.\n---\n# Demo\nUse the demo skill.",
                encoding="utf-8",
            )
            library = scan_skill_directory(tmp)

            class FakeLLM:
                def __init__(self):
                    self.calls = 0

                def create_chat_completion(self, **kwargs):
                    self.calls += 1
                    if "tools" in kwargs:
                        raise TypeError("tools are not supported")
                    if self.calls == 2:
                        return {
                            "choices": [
                                {
                                    "message": {
                                        "content": '{"tool_calls":[{"name":"skill_list","arguments":{}}]}'
                                    }
                                }
                            ]
                        }
                    return {"choices": [{"message": {"content": "Final answer"}}]}

            result = AgentRunner(FakeLLM(), seed=0, skill_library=library).run(
                [{"role": "user", "content": "Use a skill."}]
            )

            self.assertEqual(result.output, "Final answer")
            self.assertEqual(result.selected_skills, ["demo"])
            self.assertTrue(any(event.get("tool") == "skill_list" for event in result.trace))

    def test_agent_fallback_loop_executes_xml_skill_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "demo"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: demo\ndescription: Battle animation skill.\n---\n# Demo\nUse the demo skill.",
                encoding="utf-8",
            )
            library = scan_skill_directory(tmp)

            class FakeLLM:
                def __init__(self):
                    self.calls = 0

                def create_chat_completion(self, **kwargs):
                    self.calls += 1
                    if "tools" in kwargs:
                        raise TypeError("tools are not supported")
                    if self.calls == 2:
                        return {
                            "choices": [
                                {
                                    "message": {
                                        "content": """<tool_call>
<function=skill_list>
<parameter=query>
battle animation
</parameter>
</function>
</tool_call>"""
                                    }
                                }
                            ]
                        }
                    return {"choices": [{"message": {"content": "Final video prompt"}}]}

            result = AgentRunner(FakeLLM(), seed=0, skill_library=library).run(
                [{"role": "user", "content": "Make a battle animation."}]
            )

            self.assertEqual(result.output, "Final video prompt")
            self.assertTrue(any(event.get("tool") == "skill_list" for event in result.trace))

    def test_agent_context_overflow_returns_actionable_message(self):
        class FakeLLM:
            def n_ctx(self):
                return 8192

            def create_chat_completion(self, **kwargs):
                raise RuntimeError(
                    "Llama.eval: Context Shift is explicitly disabled by the C++ backend. "
                    "You MUST increase n_ctx (currently 8192) to fit the dialogue."
                )

        result = AgentRunner(FakeLLM(), seed=0).run(
            [{"role": "user", "content": "Use a long dialogue."}]
        )

        self.assertIn("Current n_ctx is 8192", result.output)
        self.assertIn("Increase Llama-cpp Model Loader n_ctx", result.output)
        self.assertTrue(result.trace[0].get("error"))

    def test_agent_auto_language_reads_chinese_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "demo"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: demo\ndescription: Demo skill.\n---\n# Demo\nEnglish content.",
                encoding="utf-8",
            )
            (skill_dir / "SKILL.cn.md").write_text("# Demo\n中文内容。", encoding="utf-8")
            library = scan_skill_directory(tmp, language="auto")

            class FakeLLM:
                def __init__(self):
                    self.calls = 0

                def create_chat_completion(self, **kwargs):
                    self.calls += 1
                    if "tools" in kwargs:
                        raise TypeError("tools are not supported")
                    if self.calls == 2:
                        return {
                            "choices": [
                                {
                                    "message": {
                                        "content": '{"tool_calls":[{"name":"skill_read","arguments":{"skill_name":"demo"}}]}'
                                    }
                                }
                            ]
                        }
                    return {"choices": [{"message": {"content": "完成"}}]}

            result = AgentRunner(FakeLLM(), seed=0, skill_library=library).run(
                [{"role": "user", "content": "请使用这个 skill。"}]
            )

            self.assertEqual(result.output, "完成")
            skill_read_events = [event for event in result.trace if event.get("tool") == "skill_read"]
            self.assertEqual(len(skill_read_events), 1)
            self.assertIn("SKILL.cn.md", skill_read_events[0]["content"])


if __name__ == "__main__":
    unittest.main()
