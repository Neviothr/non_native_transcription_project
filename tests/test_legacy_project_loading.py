from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcription_app.models import ProjectData


class LegacyProjectLoadingTests(unittest.TestCase):
    def test_removed_correction_timer_field_is_ignored(self) -> None:
        project = ProjectData.from_dict(
            {
                "metadata": {},
                "turns": [
                    {
                        "turn_id": 1,
                        "final_text": "hello",
                        "manual_correction_seconds": 12.5,
                    }
                ],
            }
        )
        self.assertEqual(len(project.turns), 1)
        self.assertEqual(project.turns[0].final_text, "hello")
        self.assertFalse(hasattr(project.turns[0], "manual_correction_seconds"))


if __name__ == "__main__":
    unittest.main()
