from __future__ import annotations

import queue
import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.gui import TranscriptionApp


class _WidgetStub:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def start(self, *args: object) -> None:
        self.calls.append(("start", *args))

    def stop(self) -> None:
        self.calls.append(("stop",))

    def configure(self, **kwargs: object) -> None:
        self.calls.append(("configure", kwargs))


class _BackgroundHarness:
    def __init__(self) -> None:
        self.progress = _WidgetStub()
        self.transcribe_button = _WidgetStub()
        self.posted: queue.Queue[object] = queue.Queue()
        self.success: list[object] = []
        self.failure: list[object] = []

    def _post_to_ui(self, callback) -> None:
        self.posted.put(callback)

    def _background_succeeded(self, result, callback, on_error=None) -> None:
        self.success.append(result)
        if callback is not None:
            callback(result)

    def _background_failed(self, exc, details, on_error=None) -> None:
        self.failure.append((exc, details))


class GuiThreadDispatchTests(unittest.TestCase):
    def test_post_to_ui_executes_immediately_on_main_thread(self) -> None:
        harness = type("Harness", (), {})()
        harness._main_thread_id = threading.get_ident()
        harness._ui_call_queue = queue.Queue()
        calls: list[str] = []

        TranscriptionApp._post_to_ui(harness, lambda: calls.append("main"))

        self.assertEqual(calls, ["main"])
        self.assertTrue(harness._ui_call_queue.empty())

    def test_post_to_ui_queues_worker_callback(self) -> None:
        harness = type("Harness", (), {})()
        harness._main_thread_id = threading.get_ident()
        harness._ui_call_queue = queue.Queue()
        calls: list[str] = []

        worker = threading.Thread(
            target=lambda: TranscriptionApp._post_to_ui(
                harness,
                lambda: calls.append("worker"),
            )
        )
        worker.start()
        worker.join(timeout=2)

        self.assertEqual(calls, [])
        callback = harness._ui_call_queue.get_nowait()
        callback()
        self.assertEqual(calls, ["worker"])

    def test_background_runner_posts_completion_without_tk_after(self) -> None:
        harness = _BackgroundHarness()
        callback_results: list[int] = []

        TranscriptionApp._run_background(
            harness,
            lambda: 42,
            callback_results.append,
        )

        deadline = time.monotonic() + 2.0
        while harness.posted.empty() and time.monotonic() < deadline:
            time.sleep(0.01)

        callback = harness.posted.get_nowait()
        callback()

        self.assertEqual(harness.success, [42])
        self.assertEqual(callback_results, [42])
        self.assertEqual(harness.failure, [])

if __name__ == "__main__":
    unittest.main()
