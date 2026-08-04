from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from transcription_app.gui import (
    _load_last_selected_files,
    _save_last_selected_files,
)


class LastSelectedFilesTests(unittest.TestCase):
    def test_selected_files_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preferences" / "last-files.json"
            selected = {
                "audio": "C:/sessions/audio.wav",
                "zoom": "C:/sessions/zoom.vtt",
                "chatgpt": "C:/sessions/chatgpt.xlsx",
                "gold": "C:/sessions/gold.xlsx",
            }

            _save_last_selected_files(path, selected)

            self.assertEqual(_load_last_selected_files(path), selected)

    def test_invalid_preferences_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last-files.json"
            path.write_text("not json", encoding="utf-8")

            self.assertEqual(_load_last_selected_files(path), {})

    def test_unknown_and_non_string_values_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last-files.json"
            path.write_text(
                json.dumps({"audio": "audio.wav", "zoom": 42, "extra": "x"}),
                encoding="utf-8",
            )

            self.assertEqual(_load_last_selected_files(path), {"audio": "audio.wav"})


if __name__ == "__main__":
    unittest.main()
