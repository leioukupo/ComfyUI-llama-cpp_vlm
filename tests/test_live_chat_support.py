import threading
import time
import tempfile
import unittest
from pathlib import Path

from support.live_chat_runtime import (
    LiveChatManager,
    STATUS_ENDED,
    build_live_session_signature,
    render_transcript,
    run_live_chat,
)
from support.skill_runtime import scan_skill_directory


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def create_chat_completion(self, **kwargs):
        if not self.replies:
            raise AssertionError("FakeLLM was called more times than expected.")
        return {"choices": [{"message": {"content": self.replies.pop(0)}}]}


class TestLiveChatRuntime(unittest.TestCase):
    def test_live_session_open_snapshot_and_clear(self):
        manager = LiveChatManager()
        signature = build_live_session_signature("demo.gguf", "hello", {"temperature": 0.7})
        session = manager.open_session("node-1", "node-1", signature, "hello")

        snapshot = session.snapshot()

        self.assertEqual(snapshot["state_uid"], "node-1")
        self.assertEqual(snapshot["status"], "idle")
        self.assertEqual(render_transcript(session.messages), "")

        manager.clear()
        self.assertIsNone(manager.get(node_id="node-1"))

    def test_live_session_waits_for_message_and_resumes(self):
        manager = LiveChatManager()
        signature = build_live_session_signature("demo.gguf", "hello", {"temperature": 0.7})
        session = manager.open_session("node-1", "node-1", signature, "hello")
        received = []

        def waiter():
            received.append(manager.wait_for_message(session, should_stop=lambda: False))

        thread = threading.Thread(target=waiter, daemon=True)
        thread.start()
        time.sleep(0.1)
        manager.queue_message(node_id="node-1", message="继续")
        thread.join(timeout=2)

        self.assertEqual(received, ["继续"])

    def test_live_run_processes_initial_and_followup_messages_until_end(self):
        manager = LiveChatManager()
        signature = build_live_session_signature("demo.gguf", "hello", {"temperature": 0.7})
        session = manager.open_session("node-1", "node-1", signature, "hello")
        events = []
        result_box = []
        assistant_events = []
        finished = threading.Event()

        def emit_state(current_session, event):
            events.append((event, current_session.status, current_session.turn_count, current_session.last_output))
            if event == "assistant":
                assistant_events.append(current_session.turn_count)
                if current_session.turn_count == 1:
                    manager.queue_message(node_id="node-1", message="第二轮")
                elif current_session.turn_count == 2:
                    manager.end_session(node_id="node-1")
                    finished.set()

        def runner():
            result_box.append(
                run_live_chat(
                    manager,
                    FakeLLM(["第一轮回复", "第二轮回复"]),
                    seed=0,
                    parameters={},
                    session=session,
                    should_stop=lambda: False,
                    emit_state=emit_state,
                    initial_user_message="第一句",
                )
            )

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        finished.wait(timeout=5)
        thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertEqual([message["role"] for message in session.messages], ["user", "assistant", "user", "assistant"])
        self.assertEqual(session.last_output, "第二轮回复")
        self.assertEqual(session.turn_count, 2)
        self.assertEqual(result_box[0].status, STATUS_ENDED)
        self.assertEqual(result_box[0].last_output, "第二轮回复")
        self.assertGreaterEqual(len(assistant_events), 2)

    def test_signature_mismatch_resets_session(self):
        manager = LiveChatManager()
        sig_a = build_live_session_signature("demo.gguf", "hello", {"temperature": 0.7})
        sig_b = build_live_session_signature("demo.gguf", "hello", {"temperature": 0.9})
        session_a = manager.open_session("node-1", "node-1", sig_a, "hello")
        session_a.append_user_message("A")

        session_b = manager.open_session("node-1", "node-1", sig_b, "hello")

        self.assertEqual(session_b.signature, sig_b)
        self.assertEqual(session_b.messages, [])
        self.assertIsNot(session_a, session_b)

    def test_live_run_can_use_skill_tools_without_polluting_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "video-prompt"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: video-prompt\ndescription: Write video prompts.\n---\n# Video Prompt\nUse cinematic beats.",
                encoding="utf-8",
            )
            library = scan_skill_directory(str(root), selected_skills="video-prompt", language="en")

            manager = LiveChatManager()
            signature = build_live_session_signature("demo.gguf", "hello", {"temperature": 0.7}, skills=library)
            session = manager.open_session("node-1", "node-1", signature, "hello")

            def emit_state(current_session, event):
                if event == "assistant":
                    manager.end_session(node_id="node-1")

            result = run_live_chat(
                manager,
                FakeLLM(
                    [
                        '{"tool_calls":[{"name":"skill_read","arguments":{"skill_name":"video-prompt"}}]}',
                        "最终提示词：小男孩与怪兽在雨夜街巷中对峙。",
                    ]
                ),
                seed=0,
                parameters={},
                session=session,
                should_stop=lambda: False,
                emit_state=emit_state,
                initial_user_message="生成视频提示词",
                skill_library=library,
            )

            transcript = render_transcript(result.messages)

            self.assertEqual(result.status, STATUS_ENDED)
            self.assertEqual(result.selected_skills, ["video-prompt"])
            self.assertIn("最终提示词", result.last_output)
            self.assertIn("最终提示词", transcript)
            self.assertNotIn("Tool results:", transcript)


if __name__ == "__main__":
    unittest.main()
