from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.gui import TranscriptionApp, _training_record_summary


class EvaluationTabLoggingTests(unittest.TestCase):
    def test_training_record_summary_counts_valid_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quality_training.json"
            path.write_text(
                json.dumps(
                    [
                        {"features": [0.1, 0.2], "label": 0},
                        {"features": [0.3, 0.4], "label": 1},
                        {"features": [0.5, 0.6], "label": 1},
                        {"features": "invalid", "label": 2},
                        {"missing": "fields"},
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(_training_record_summary(path), (3, {0: 1, 1: 2}))

    def test_evaluation_tab_has_separate_process_log(self) -> None:
        source = inspect.getsource(TranscriptionApp._build_evaluation_tab)
        self.assertIn('text="Evaluation results"', source)
        self.assertIn('text="Process log"', source)
        self.assertIn("self.evaluation_log = tk.Text", source)
        self.assertIn('font=("Consolas", 9)', source)
        self.assertIn("self.evaluation_timer_var = tk.StringVar", source)
        self.assertIn("Process time: 00:00:00.0", source)

    def test_process_log_has_live_elapsed_timer(self) -> None:
        append_source = inspect.getsource(TranscriptionApp._append_evaluation_log)
        begin_source = inspect.getsource(TranscriptionApp._begin_evaluation_operation)
        finish_source = inspect.getsource(TranscriptionApp._finish_evaluation_operation)
        fail_source = inspect.getsource(TranscriptionApp._fail_evaluation_operation)
        update_source = inspect.getsource(TranscriptionApp._update_evaluation_timer)

        self.assertIn("elapsed_prefix", append_source)
        self.assertIn("_format_elapsed(elapsed)", append_source)
        self.assertIn("self._start_evaluation_timer(name)", begin_source)
        self.assertIn("self._stop_evaluation_timer(started_at, outcome)", finish_source)
        self.assertIn('self._stop_evaluation_timer(started_at, "failed")', fail_source)
        self.assertIn("self.after(", update_source)
        self.assertIn("100, self._update_evaluation_timer", update_source)

    def test_all_requested_actions_start_tab4_logging(self) -> None:
        expected = {
            "calculate_evaluation": "Calculate Evaluation",
            "add_training_examples": "Add Gold Examples",
            "train_models": "Train and Compare ML Models",
            "export_report": "Export HTML Report",
            "export_excel": "Export Excel",
        }
        for method_name, operation_name in expected.items():
            with self.subTest(method=method_name):
                source = inspect.getsource(getattr(TranscriptionApp, method_name))
                self.assertIn(f'operation = "{operation_name}"', source)
                self.assertIn("self._begin_evaluation_operation(operation)", source)

    def test_background_runner_accepts_error_log_callback(self) -> None:
        source = inspect.getsource(TranscriptionApp._run_background)
        self.assertIn("on_error=None", source)
        self.assertIn("self._background_failed", source)


if __name__ == "__main__":
    unittest.main()
