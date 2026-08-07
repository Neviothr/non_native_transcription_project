from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.models import ProjectData, ProjectMetadata, SpeechEvent, Turn
from transcription_app.gui import (
    TranscriptionApp,
    _speech_delay_update_from_details,
)
from transcription_app.speech_events import (
    AUTOMATIC_DELAY_SOURCE,
    reassociate_automatic_delay_events,
    remap_nonautomatic_event_turn_ids,
    render_turn_with_speech_delays,
    replace_detected_delay_events,
)
from transcription_app.workflow import (
    analyze_turns,
    apply_speaker_mapping,
    propagate_detected_learner_identity,
)


def _delay(start: float, end: float) -> dict[str, object]:
    return {
        "interval_index": 0,
        "interval_start_seconds": 0.0,
        "interval_end_seconds": 10.0,
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": end - start,
        "loudest_frame_dbfs": None,
        "event_type": "silent_pause",
    }


class _Variable:
    def __init__(self, value: object) -> None:
        self.value = value

    def get(self) -> object:
        return self.value

    def set(self, value: object) -> None:
        self.value = value


class _TextValue:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self, _start: str, _end: str) -> str:
        return self.value


class _AudioPlayer:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object, object]] = []

    def play(self, path: object, start: object, end: object) -> float:
        self.calls.append((path, start, end))
        return 0.1


class SpeechEventAssociationTests(unittest.TestCase):
    def test_internal_pause_is_timed_and_rendered_without_mutating_literal_text(self) -> None:
        turn = Turn(
            turn_id=1,
            start=0.0,
            end=4.0,
            speaker="Learner",
            final_text="I want to explain this",
        )
        project = ProjectData(turns=[turn])

        added = replace_detected_delay_events(project, [_delay(1.8, 2.4)])
        rendered = render_turn_with_speech_delays(project, turn)

        self.assertEqual(len(added), 1)
        self.assertEqual(added[0].event_type, "silent_pause")
        self.assertEqual(added[0].turn_id, 1)
        self.assertIn("[pause 0.60s]", rendered)
        self.assertEqual(turn.final_text, "I want to explain this")

    def test_empty_final_text_never_resurrects_model_words(self) -> None:
        turn = Turn(
            turn_id=1,
            start=0.0,
            end=4.0,
            model_text="obsolete model wording",
            final_text="",
        )
        project = ProjectData(turns=[turn])
        replace_detected_delay_events(project, [_delay(1.8, 2.4)])

        rendered = render_turn_with_speech_delays(project, turn)

        self.assertEqual(rendered, "[pause 0.60s]")
        self.assertNotIn("obsolete", rendered)

    def test_gap_between_different_known_speakers_is_response_gap(self) -> None:
        project = ProjectData(
            turns=[
                Turn(1, start=0.0, end=1.0, speaker="Learner", final_text="Yes"),
                Turn(2, start=2.0, end=3.0, speaker="Teacher", final_text="Why"),
            ]
        )

        [event] = replace_detected_delay_events(project, [_delay(1.2, 1.8)])

        self.assertEqual(event.event_type, "response_gap")
        self.assertIsNone(event.turn_id)
        self.assertEqual(render_turn_with_speech_delays(project, project.turns[0]), "Yes")
        self.assertEqual(
            render_turn_with_speech_delays(project, project.turns[1]),
            "[response gap 0.60s] Why",
        )

    def test_response_gap_is_included_in_following_turn_playback(self) -> None:
        project = ProjectData(
            metadata=ProjectMetadata(audio_file="recording.wav"),
            turns=[
                Turn(1, start=0.0, end=1.0, speaker="Student", final_text="Yes"),
                Turn(2, start=2.0, end=3.0, speaker="Teacher", final_text="Why"),
            ],
        )
        replace_detected_delay_events(project, [_delay(1.2, 1.8)])
        player = _AudioPlayer()
        app = SimpleNamespace(
            project=project,
            playing_turn_index=None,
            audio_var=_Variable("recording.wav"),
            notebook=SimpleNamespace(select=lambda _tab: None),
            project_tab=object(),
            turn_audio_player=player,
            playback_after_id=None,
            after=lambda _milliseconds, _callback: "after-id",
            _playback_finished=lambda: None,
            refresh_turn_table=lambda: None,
            _set_status=lambda _message: None,
        )

        TranscriptionApp.play_turn_audio(app, 1)

        self.assertEqual(player.calls, [("recording.wav", 1.2, 3.0)])

    def test_pause_inside_long_overlapping_turn_is_not_a_response_gap(self) -> None:
        long_turn = Turn(
            1,
            start=0.0,
            end=10.0,
            speaker="Teacher",
            final_text="A long explanation",
        )
        overlapping_turn = Turn(
            2,
            start=4.0,
            end=5.0,
            speaker="Student",
            final_text="Yes",
        )
        project = ProjectData(turns=[long_turn, overlapping_turn])

        [event] = replace_detected_delay_events(project, [_delay(6.5, 7.0)])

        self.assertEqual(event.event_type, "silent_pause")
        self.assertEqual(event.turn_id, long_turn.turn_id)

    def test_pause_crossing_a_known_speaker_boundary_is_response_gap(self) -> None:
        project = ProjectData(
            turns=[
                Turn(1, start=0.0, end=2.0, speaker="Learner", final_text="Yes"),
                Turn(2, start=2.0, end=4.0, speaker="Teacher", final_text="Why"),
            ]
        )

        [event] = replace_detected_delay_events(project, [_delay(1.8, 2.2)])

        self.assertEqual(event.event_type, "response_gap")
        self.assertIsNone(event.turn_id)
        self.assertEqual(event.details["association"], "speaker_boundary")

    def test_gap_without_speaker_change_is_preserved_after_preceding_turn(self) -> None:
        project = ProjectData(
            turns=[
                Turn(1, start=0.0, end=1.0, speaker="Unknown", final_text="I think"),
                Turn(2, start=2.0, end=3.0, speaker="Unknown", final_text="maybe"),
            ]
        )

        [event] = replace_detected_delay_events(project, [_delay(1.2, 1.8)])

        self.assertEqual(event.event_type, "silent_pause")
        self.assertEqual(event.turn_id, 1)
        self.assertEqual(
            render_turn_with_speech_delays(project, project.turns[0]),
            "I think [pause 0.60s]",
        )

    def test_rerun_replaces_only_automatic_delay_events(self) -> None:
        manual = SpeechEvent(
            event_id=20,
            turn_id=1,
            event_type="filled_pause",
            start=0.2,
            end=0.4,
            text="um",
            source="manual",
        )
        stale = SpeechEvent(
            event_id=21,
            turn_id=1,
            event_type="silent_pause",
            start=0.5,
            end=1.0,
            source=AUTOMATIC_DELAY_SOURCE,
        )
        project = ProjectData(
            turns=[Turn(1, start=0.0, end=3.0, final_text="hello there")],
            speech_events=[manual, stale],
        )

        replace_detected_delay_events(project, [_delay(1.2, 1.6)])

        self.assertIn(manual, project.speech_events)
        automatic = [
            event
            for event in project.speech_events
            if event.source == AUTOMATIC_DELAY_SOURCE
        ]
        self.assertEqual(len(automatic), 1)
        self.assertAlmostEqual(automatic[0].start, 1.2)

    def test_manual_event_turn_references_follow_turn_id_remapping(self) -> None:
        event = SpeechEvent(
            event_id=1,
            turn_id=3,
            event_type="revision",
            start=None,
            end=None,
            source="manual",
            details={"previous_turn_id": 2, "following_turn_id": 3},
        )
        project = ProjectData(speech_events=[event])

        remap_nonautomatic_event_turn_ids(project, {2: 1, 3: 2})

        self.assertEqual(event.turn_id, 2)
        self.assertEqual(event.details["previous_turn_id"], 1)
        self.assertEqual(event.details["following_turn_id"], 2)

    def test_detector_failure_preserves_only_matching_audio_evidence(self) -> None:
        source_path = str((ROOT / "same-audio.wav").resolve())
        event = SpeechEvent(
            event_id=1,
            turn_id=1,
            event_type="silent_pause",
            start=1.0,
            end=1.5,
            source=AUTOMATIC_DELAY_SOURCE,
            details={
                "audio_source_path": source_path,
                "audio_source_size_bytes": 1234,
                "audio_source_modified_time_ns": 5678,
            },
        )
        project = ProjectData(speech_events=[event])
        failed_details: dict[str, object] = {
            "speech_delay_events": [],
            "speech_delay_detection_error": "transient failure",
            "audio_source_path": source_path,
            "audio_source_size_bytes": 1234,
            "audio_source_modified_time_ns": 5678,
        }

        self.assertIsNone(
            _speech_delay_update_from_details(project, failed_details)
        )

        failed_details["audio_source_size_bytes"] = 9999
        self.assertEqual(
            _speech_delay_update_from_details(project, failed_details),
            [],
        )

        failed_details["speech_delay_detection_error"] = ""
        self.assertEqual(
            _speech_delay_update_from_details(project, failed_details),
            [],
        )

    def test_reassociation_uses_corrected_speaker_labels(self) -> None:
        first = Turn(1, start=0.0, end=1.0, speaker="Unknown", final_text="Yes")
        second = Turn(2, start=2.0, end=3.0, speaker="Unknown", final_text="Why")
        project = ProjectData(turns=[first, second])
        [event] = replace_detected_delay_events(project, [_delay(1.2, 1.8)])
        self.assertEqual(event.event_type, "silent_pause")

        first.speaker = "Learner"
        second.speaker = "Teacher"
        reassociate_automatic_delay_events(project)

        self.assertEqual(event.event_type, "response_gap")
        self.assertIsNone(event.turn_id)

    def test_applying_speaker_mapping_reassociates_delay_events(self) -> None:
        first = Turn(
            1,
            start=0.0,
            end=1.0,
            speaker_raw="speaker-a",
            speaker="Unknown",
            final_text="Yes",
        )
        second = Turn(
            2,
            start=2.0,
            end=3.0,
            speaker_raw="speaker-b",
            speaker="Unknown",
            final_text="Why",
        )
        project = ProjectData(turns=[first, second])
        [event] = replace_detected_delay_events(project, [_delay(1.2, 1.8)])
        self.assertEqual(event.event_type, "silent_pause")

        apply_speaker_mapping(
            project,
            {"speaker-a": "Student", "speaker-b": "Teacher"},
        )

        self.assertEqual(event.event_type, "response_gap")
        self.assertIsNone(event.turn_id)
        self.assertFalse(event.details["token_position_estimated"])
        self.assertEqual(event.details["previous_turn_id"], 1)
        self.assertEqual(event.details["following_turn_id"], 2)

    def test_new_silent_pause_owner_is_put_in_manual_review_queue(self) -> None:
        first = Turn(
            1,
            start=0.0,
            end=1.0,
            speaker="Student",
            final_text="Yes",
            hesitation_or_repetition=False,
            manual_review=False,
        )
        second = Turn(
            2,
            start=2.0,
            end=3.0,
            speaker="Teacher",
            final_text="Why",
            hesitation_or_repetition=False,
            manual_review=False,
        )
        project = ProjectData(turns=[first, second])
        [event] = replace_detected_delay_events(project, [_delay(1.2, 1.8)])
        self.assertEqual(event.event_type, "response_gap")

        second.speaker = "Student"
        reassociate_automatic_delay_events(project)

        self.assertEqual(event.event_type, "silent_pause")
        self.assertEqual(event.turn_id, first.turn_id)
        self.assertTrue(first.hesitation_or_repetition)
        self.assertTrue(first.manual_review)

    def test_unchanged_pause_does_not_override_reviewer_flags(self) -> None:
        turn = Turn(
            1,
            start=0.0,
            end=3.0,
            speaker="Student",
            final_text="I agree",
            hesitation_or_repetition=False,
            manual_review=False,
        )
        project = ProjectData(turns=[turn])
        [event] = replace_detected_delay_events(project, [_delay(1.2, 1.8)])
        event.reviewed = True
        turn.hesitation_or_repetition = False
        turn.manual_review = False

        reassociate_automatic_delay_events(project)

        self.assertFalse(turn.hesitation_or_repetition)
        self.assertFalse(turn.manual_review)

    def test_changed_delay_location_invalidates_prior_confirmation(self) -> None:
        first = Turn(1, start=0.0, end=1.0, speaker="Student", final_text="Yes")
        second = Turn(2, start=2.0, end=3.0, speaker="Teacher", final_text="Why")
        project = ProjectData(turns=[first, second])
        [event] = replace_detected_delay_events(project, [_delay(1.2, 1.8)])
        event.reviewed = True

        second.speaker = "Student"
        invalidated = reassociate_automatic_delay_events(project)

        self.assertEqual(event.event_type, "silent_pause")
        self.assertFalse(event.reviewed)
        self.assertTrue(first.manual_review)
        self.assertEqual(invalidated, {event.event_id})

    def test_boundary_reclassification_cannot_reuse_old_confirmation(self) -> None:
        first = Turn(1, start=0.0, end=4.0, speaker="Student", final_text="Long")
        second = Turn(2, start=4.0, end=5.0, speaker="Teacher", final_text="Short")
        project = ProjectData(turns=[first, second])
        [event] = replace_detected_delay_events(project, [_delay(3.8, 4.2)])
        self.assertEqual(event.event_type, "response_gap")
        event.reviewed = True

        second.speaker = "Student"
        invalidated = reassociate_automatic_delay_events(project)

        self.assertEqual(event.event_type, "silent_pause")
        self.assertEqual(event.turn_id, second.turn_id)
        self.assertFalse(event.reviewed)
        self.assertEqual(invalidated, {event.event_id})

    def test_confirmed_delay_no_longer_forces_high_quality_turn_review(self) -> None:
        turn = Turn(
            1,
            start=0.0,
            end=3.0,
            speaker="Student",
            zoom_text="I agree",
            chatgpt_text="I agree",
            model_text="I agree",
            final_text="I agree",
            quality_target_text="I agree",
            model_confidence=0.99,
            manual_review=False,
        )
        project = ProjectData(turns=[turn])
        [event] = replace_detected_delay_events(project, [_delay(1.2, 1.8)])

        analyze_turns(project)
        self.assertTrue(turn.manual_review)
        self.assertTrue(turn.hesitation_or_repetition)

        event.reviewed = True
        analyze_turns(project)

        self.assertFalse(turn.manual_review)
        self.assertFalse(turn.hesitation_or_repetition)

    def test_gui_save_cannot_clear_review_while_delay_is_unconfirmed(self) -> None:
        turn = Turn(
            1,
            start=0.0,
            end=3.0,
            speaker_raw="Student",
            speaker="Student",
            final_text="I agree",
            manual_review=False,
        )
        project = ProjectData(turns=[turn])
        [event] = replace_detected_delay_events(project, [_delay(1.2, 1.8)])
        app = SimpleNamespace(
            project=project,
            current_turn_index=0,
            _loading_editor=False,
            _saving_editor=False,
            editor_speaker_var=_Variable("Student"),
            hebrew_var=_Variable(False),
            hesitation_var=_Variable(False),
            self_correction_var=_Variable(False),
            unclear_var=_Variable(False),
            overlap_var=_Variable(False),
            manual_review_var=_Variable(False),
            delay_reviewed_var=_Variable(False),
            final_text=_TextValue("I agree"),
            refresh_turn_table=lambda: None,
            _set_status=lambda _message: None,
        )

        TranscriptionApp.save_editor_to_turn(
            app,
            silent=True,
            refresh_table=False,
        )

        self.assertFalse(event.reviewed)
        self.assertTrue(turn.manual_review)
        self.assertTrue(app.manual_review_var.get())

    def test_gui_delay_confirmation_can_release_manual_review_flag(self) -> None:
        turn = Turn(
            1,
            start=0.0,
            end=3.0,
            speaker_raw="Student",
            speaker="Student",
            final_text="I agree",
            manual_review=True,
        )
        project = ProjectData(turns=[turn])
        [event] = replace_detected_delay_events(project, [_delay(1.2, 1.8)])
        app = SimpleNamespace(
            project=project,
            current_turn_index=0,
            _loading_editor=False,
            _saving_editor=False,
            editor_speaker_var=_Variable("Student"),
            hebrew_var=_Variable(False),
            hesitation_var=_Variable(False),
            self_correction_var=_Variable(False),
            unclear_var=_Variable(False),
            overlap_var=_Variable(False),
            manual_review_var=_Variable(False),
            delay_reviewed_var=_Variable(True),
            final_text=_TextValue("I agree"),
            refresh_turn_table=lambda: None,
            _set_status=lambda _message: None,
        )

        TranscriptionApp.save_editor_to_turn(
            app,
            silent=True,
            refresh_table=False,
        )

        self.assertTrue(event.reviewed)
        self.assertFalse(turn.manual_review)
        self.assertFalse(app.manual_review_var.get())

    def test_propagated_learner_name_reclassifies_response_gap(self) -> None:
        first = Turn(
            1,
            start=0.0,
            end=1.0,
            speaker_raw="Unknown",
            speaker="Unknown",
            final_text="My name is Dana",
        )
        second = Turn(
            2,
            start=2.0,
            end=3.0,
            speaker_raw="Teacher",
            speaker="Teacher",
            final_text="Welcome",
        )
        project = ProjectData(
            metadata=ProjectMetadata(conversation_type="Human teacher"),
            turns=[first, second],
        )
        [event] = replace_detected_delay_events(project, [_delay(1.2, 1.8)])
        self.assertEqual(event.event_type, "silent_pause")

        changed = propagate_detected_learner_identity(project)

        self.assertEqual(changed, 1)
        self.assertEqual(first.speaker, "Dana")
        self.assertEqual(event.event_type, "response_gap")
        self.assertIsNone(event.turn_id)


if __name__ == "__main__":
    unittest.main()
