from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcription_app.models import ProjectData, TranscriptSegment, Turn
from transcription_app.quality import extract_features
from transcription_app.workflow import (
    align_all_sources,
    analyze_turns,
    training_examples_from_project,
)


class SegmentBoundaryQualityTests(unittest.TestCase):
    def _project(self) -> ProjectData:
        return ProjectData(
            turns=[
                Turn(
                    turn_id=1,
                    start=0.0,
                    end=2.0,
                    speaker_raw="Student",
                    speaker="Student",
                    model_text="I like electrical engineering",
                ),
                Turn(
                    turn_id=2,
                    start=2.0,
                    end=4.0,
                    speaker_raw="Student",
                    speaker="Student",
                    model_text="because it is interesting",
                ),
            ],
            source_transcripts={
                "zoom": [
                    TranscriptSegment(
                        0.0,
                        4.0,
                        "Student",
                        "I like electrical engineering because it is interesting",
                    )
                ],
                "chatgpt": [
                    TranscriptSegment(
                        0.0,
                        1.5,
                        "Student",
                        "I like electrical",
                    ),
                    TranscriptSegment(
                        1.5,
                        3.0,
                        "Student",
                        "engineering because it",
                    ),
                    TranscriptSegment(
                        3.0,
                        4.0,
                        "Student",
                        "is interesting",
                    ),
                ],
                "gold": [
                    TranscriptSegment(
                        0.0,
                        4.0,
                        "Student",
                        "I like electrical engineering because it is interesting",
                    )
                ],
            },
        )

    def test_source_segmentation_does_not_reduce_quality(self) -> None:
        project = self._project()

        align_all_sources(project)

        for turn in project.turns:
            features = extract_features(turn)
            self.assertAlmostEqual(features["agreement"], 1.0)
            self.assertAlmostEqual(features["word_disagreement_rate"], 0.0)
        analyze_turns(project)

        self.assertEqual(
            [turn.quality_label for turn in project.turns],
            ["Transcript acceptable", "Transcript acceptable"],
        )

    def test_gold_training_labels_use_split_gold_text(self) -> None:
        project = self._project()

        align_all_sources(project)
        _rows, labels = training_examples_from_project(project)

        self.assertEqual(labels, [0, 0])
        self.assertEqual(
            [turn.gold_text for turn in project.turns],
            [
                "I like electrical engineering",
                "because it is interesting",
            ],
        )


if __name__ == "__main__":
    unittest.main()
