import asyncio
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from support.agent_runtime import (
    AgentRunner,
    ConversationSession,
    STATUS_AWAITING_USER,
    build_session_signature,
    extract_tool_calls,
    looks_like_user_question,
    normalize_session,
    strip_thinking,
)
from support.auto_budget import normalize_n_ctx_for_chat_handler, resolve_auto_budget
from support.mcp_runtime import MCPRuntimeConfig, MCPServerConfig, MCPToolbox, format_mcp_result, parse_mcp_config
from support.skill_runtime import scan_skill_directory


class TestAutoBudget(unittest.TestCase):
    def _write_fake_gguf(self, path, size_gb=20):
        metadata = {
            "general.architecture": "qwen3",
            "qwen3.block_count": 48,
            "qwen3.attention.head_count": 32,
            "qwen3.attention.head_count_kv": 8,
            "qwen3.embedding_length": 4096,
            "qwen3.attention.key_length": 128,
            "qwen3.attention.value_length": 128,
        }
        with open(path, "wb") as f:
            f.write(b"GGUF")
            f.write(struct.pack("<I", 3))
            f.write(struct.pack("<Q", 0))
            f.write(struct.pack("<Q", len(metadata)))
            for key, value in metadata.items():
                key_bytes = key.encode("utf-8")
                f.write(struct.pack("<Q", len(key_bytes)))
                f.write(key_bytes)
                if isinstance(value, str):
                    value_bytes = value.encode("utf-8")
                    f.write(struct.pack("<I", 8))
                    f.write(struct.pack("<Q", len(value_bytes)))
                    f.write(value_bytes)
                else:
                    f.write(struct.pack("<I", 4))
                    f.write(struct.pack("<I", value))
            f.truncate(size_gb * 1024 ** 3)

    def test_qwen35_auto_budget_uses_large_context_and_keeps_retry_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "qwen.gguf"
            mmproj = root / "mmproj.gguf"
            self._write_fake_gguf(model, size_gb=20)
            mmproj.write_bytes(b"x")
            with open(mmproj, "ab") as f:
                f.truncate(1024 ** 3)

            plan = resolve_auto_budget(
                model_path=str(model),
                mmproj_path=str(mmproj),
                chat_handler="Qwen3.5",
                n_ctx=-1,
                vram_limit=-1,
                auto_max_ctx=65536,
                is_qwen35_family=True,
                cuda_memory=(23.0, 24.0),
            )

            self.assertTrue(plan.auto_n_ctx)
            self.assertTrue(plan.auto_vram)
            self.assertGreaterEqual(plan.n_ctx, 32768)
            self.assertLessEqual(plan.n_ctx, 65536)
            self.assertGreater(plan.n_gpu_layers, 0)
            self.assertEqual(plan.attempts[0].n_ctx, plan.n_ctx)
            self.assertEqual(plan.attempts[-1].n_ctx, 16384)

    def test_auto_context_respects_node_max_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "qwen.gguf"
            mmproj = root / "mmproj.gguf"
            self._write_fake_gguf(model, size_gb=20)
            mmproj.write_bytes(b"x")
            with open(mmproj, "ab") as f:
                f.truncate(1024 ** 3)

            plan = resolve_auto_budget(
                model_path=str(model),
                mmproj_path=str(mmproj),
                chat_handler="Qwen3.5",
                n_ctx=-1,
                vram_limit=-1,
                auto_max_ctx=32768,
                is_qwen35_family=True,
                cuda_memory=(23.0, 24.0),
            )

            self.assertLessEqual(plan.n_ctx, 32768)

    def test_manual_qwen35_values_are_clamped_to_safe_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "qwen.gguf"
            mmproj = root / "mmproj.gguf"
            self._write_fake_gguf(model, size_gb=20)
            mmproj.write_bytes(b"x")
            with open(mmproj, "ab") as f:
                f.truncate(1024 ** 3)

            plan = resolve_auto_budget(
                model_path=str(model),
                mmproj_path=str(mmproj),
                chat_handler="Qwen3.5",
                n_ctx=35967,
                vram_limit=89,
                auto_max_ctx=32768,
                is_qwen35_family=True,
                cuda_memory=(23.0, 24.0),
            )

            self.assertFalse(plan.auto_n_ctx)
            self.assertFalse(plan.auto_vram)
            self.assertLessEqual(plan.n_ctx, 32768)
            self.assertGreater(plan.n_gpu_layers, 0)
            self.assertEqual(plan.attempts[0].n_ctx, 32768)

    def test_auto_context_honors_manual_vram_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "qwen.gguf"
            mmproj = root / "mmproj.gguf"
            self._write_fake_gguf(model, size_gb=20)
            mmproj.write_bytes(b"x")
            with open(mmproj, "ab") as f:
                f.truncate(1024 ** 3)

            plan = resolve_auto_budget(
                model_path=str(model),
                mmproj_path=str(mmproj),
                chat_handler="Qwen3.5",
                n_ctx=-1,
                vram_limit=18,
                auto_max_ctx=65536,
                is_qwen35_family=True,
                cuda_memory=(23.0, 24.0),
            )

            self.assertTrue(plan.auto_n_ctx)
            self.assertFalse(plan.auto_vram)
            self.assertLessEqual(plan.n_ctx, 65536)
            self.assertGreater(plan.n_gpu_layers, 10)

    def test_qwen35_manual_context_is_clamped_but_auto_sentinel_survives(self):
        handlers = {"Qwen3.5"}
        self.assertEqual(normalize_n_ctx_for_chat_handler("Qwen3.5", 8192, handlers), 16384)
        self.assertEqual(normalize_n_ctx_for_chat_handler("Qwen3.5", -1, handlers), -1)


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

    def test_agent_native_ask_user_enters_awaiting_state(self):
        class FakeLLM:
            def create_chat_completion(self, **kwargs):
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_question",
                                        "type": "function",
                                        "function": {
                                            "name": "ask_user",
                                            "arguments": json.dumps(
                                                {
                                                    "question": "你希望哪种风格？",
                                                    "fields": ["style", "duration"],
                                                },
                                                ensure_ascii=False,
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }

        result = AgentRunner(FakeLLM(), seed=0).run(
            [{"role": "user", "content": "小男孩和怪兽战斗"}]
        )

        self.assertEqual(result.status, STATUS_AWAITING_USER)
        self.assertEqual(result.output, "你希望哪种风格？")
        self.assertEqual(result.pending_question, "你希望哪种风格？")
        self.assertEqual(result.pending_fields, ["style", "duration"])
        self.assertTrue(result.pending_question_id)
        self.assertEqual(result.trace[-1]["tool_kind"], "clarification")
        self.assertEqual(result.trace[-1]["source"], "native")
        self.assertEqual(result.transcript[-1]["content"], "你希望哪种风格？")

    def test_agent_json_fallback_ask_user_enters_awaiting_state(self):
        class FakeLLM:
            def __init__(self):
                self.calls = 0

            def create_chat_completion(self, **kwargs):
                self.calls += 1
                if "tools" in kwargs:
                    raise TypeError("tools are not supported")
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "tool_calls": [
                                            {
                                                "name": "ask_user",
                                                "arguments": {"question": "请选择视频风格。"},
                                            }
                                        ]
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ]
                }

        result = AgentRunner(FakeLLM(), seed=0).run(
            [{"role": "user", "content": "小男孩和怪兽战斗"}]
        )

        self.assertEqual(result.status, STATUS_AWAITING_USER)
        self.assertEqual(result.pending_question, "请选择视频风格。")
        self.assertEqual(result.trace[-1]["source"], "fallback")

    def test_agent_xml_fallback_ask_user_enters_awaiting_state(self):
        class FakeLLM:
            def create_chat_completion(self, **kwargs):
                if "tools" in kwargs:
                    raise TypeError("tools are not supported")
                return {
                    "choices": [
                        {
                            "message": {
                                "content": """<tool_call>
<function=ask_user>
<parameter=question>
你希望哪种风格？
</parameter>
<parameter=fields>
["style"]
</parameter>
</function>
</tool_call>"""
                            }
                        }
                    ]
                }

        result = AgentRunner(FakeLLM(), seed=0).run(
            [{"role": "user", "content": "小男孩和怪兽战斗"}]
        )

        self.assertEqual(result.status, STATUS_AWAITING_USER)
        self.assertEqual(result.pending_question, "你希望哪种风格？")
        self.assertEqual(result.pending_fields, ["style"])

    def test_natural_language_question_detection(self):
        self.assertTrue(looks_like_user_question("你希望哪种风格？"))
        self.assertTrue(looks_like_user_question("Which style would you like?"))
        self.assertFalse(looks_like_user_question("Here is the final prompt for a 15 second video."))

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
            self.assertEqual(result.selected_skills, [])
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

    def test_agent_native_tool_calls_execute_and_update_transcript(self):
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
                    if self.calls == 1:
                        return {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": "",
                                        "tool_calls": [
                                            {
                                                "id": "call_1",
                                                "type": "function",
                                                "function": {
                                                    "name": "skill_read",
                                                    "arguments": '{"skill_name":"demo"}',
                                                },
                                            }
                                        ],
                                    }
                                }
                            ]
                        }
                    return {"choices": [{"message": {"role": "assistant", "content": "Final answer"}}]}

            result = AgentRunner(FakeLLM(), seed=0, skill_library=library).run(
                [{"role": "user", "content": "Use a skill."}]
            )

            self.assertEqual(result.output, "Final answer")
            self.assertEqual(result.selected_skills, ["demo"])
            self.assertEqual(result.trace[0]["branch"], "native")
            self.assertTrue(any(message.get("role") == "tool" for message in result.transcript))

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
            self.assertEqual(result.selected_skills, ["demo"])
            skill_read_events = [event for event in result.trace if event.get("tool") == "skill_read"]
            self.assertEqual(len(skill_read_events), 1)
            self.assertIn("SKILL.cn.md", skill_read_events[0]["content"])

    def test_agent_transcript_preserves_tool_loop_without_system_prompt(self):
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
                                        "content": '{"tool_calls":[{"name":"skill_read","arguments":{"skill_name":"demo"}}]}'
                                    }
                                }
                            ]
                        }
                    return {"choices": [{"message": {"content": "Final answer"}}]}

            result = AgentRunner(FakeLLM(), seed=0, skill_library=library).run(
                [
                    {"role": "system", "content": "System prompt"},
                    {"role": "user", "content": "Use a skill."},
                ]
            )

            self.assertEqual(result.selected_skills, ["demo"])
            self.assertEqual(result.transcript[0]["role"], "user")
            self.assertFalse(any(message.get("role") == "system" for message in result.transcript))
            self.assertTrue(any("Tool results" in str(message.get("content")) for message in result.transcript))

    def test_conversation_session_roundtrip_and_signature(self):
        signature = build_session_signature({"model": "a.gguf", "system_prompt": "hi"})
        session = ConversationSession(
            state_uid="42",
            signature=signature,
            messages=[{"role": "user", "content": "hello"}],
            turn_count=1,
            last_output="world",
            status=STATUS_AWAITING_USER,
            pending_question="Which style?",
            pending_fields=["style"],
            pending_question_id="abc123",
        )

        restored = normalize_session(session.to_dict())

        self.assertTrue(restored.compatible(signature))
        self.assertEqual(restored.messages[0]["content"], "hello")
        self.assertEqual(restored.turn_count, 1)
        self.assertEqual(restored.status, STATUS_AWAITING_USER)
        self.assertEqual(restored.pending_question, "Which style?")
        self.assertEqual(restored.pending_fields, ["style"])
        self.assertEqual(restored.pending_question_id, "abc123")
        self.assertFalse(restored.compatible(build_session_signature({"model": "b.gguf"})))


if __name__ == "__main__":
    unittest.main()
