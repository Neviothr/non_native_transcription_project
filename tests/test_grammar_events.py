from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.grammar_events import (
    GRAMMAR_EVENT_TYPE,
    GRAMMAR_GUARD_SOURCE,
    grammar_events_for_turn,
    refresh_grammar_preservation_events,
    set_grammar_events_confirmed,
)
from transcription_app.gui import TranscriptionApp
from transcription_app.models import ProjectData, ProjectMetadata, Turn
from transcription_app.workflow import analyze_turns, choose_initial_text


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


class _AlwaysAcceptablePredictor:
    def predict_proba(self, _features: object) -> tuple[float, float, float]:
        return (1.0, 0.0, 0.0)


def _source_disagreement_turn(
    *,
    speaker: str = "Student",
    manual_review: bool = False,
) -> Turn:
    return Turn(
        turn_id=1,
        speaker_raw=speaker,
        speaker=speaker,
        zoom_text="I saw a enemy today",
        chatgpt_text="I saw an enemy today",
        model_text="I saw an enemy today",
        final_text="I saw an enemy today",
        quality_target_text="I saw an enemy today",
        model_confidence=0.99,
        manual_review=manual_review,
    )


def _editor_stub(
    project: ProjectData,
    *,
    final_text: str,
    grammar_confirmed: bool,
    manual_review: bool = False,
) -> SimpleNamespace:
    turn = project.turns[0]
    return SimpleNamespace(
        project=project,
        current_turn_index=0,
        _loading_editor=False,
        _saving_editor=False,
        editor_speaker_var=_Variable(turn.speaker),
        hebrew_var=_Variable(False),
        hesitation_var=_Variable(False),
        self_correction_var=_Variable(False),
        unclear_var=_Variable(False),
        overlap_var=_Variable(False),
        manual_review_var=_Variable(manual_review),
        delay_reviewed_var=_Variable(False),
        grammar_reviewed_var=_Variable(grammar_confirmed),
        final_text=_TextValue(final_text),
        refresh_turn_table=lambda: None,
        _set_status=lambda _message: None,
    )


class GrammarEventWorkflowTests(unittest.TestCase):
    def test_minority_learner_wording_survives_source_voting_as_review_evidence(self) -> None:
        turn = Turn(
            turn_id=1,
            speaker_raw="Student",
            speaker="Student",
            zoom_text="Do you have a enemy?",
            chatgpt_text="Do you have an enemy?",
            model_text="Do you have an enemy?",
            model_confidence=0.99,
            manual_review=False,
        )
        turn.final_text = choose_initial_text(turn)
        original_final = turn.final_text
        project = ProjectData(turns=[turn])

        analyze_turns(project)

        [event] = grammar_events_for_turn(project, turn.turn_id)
        self.assertEqual(original_final, "Do you have an enemy?")
        self.assertEqual(turn.final_text, original_final)
        self.assertEqual(event.event_type, GRAMMAR_EVENT_TYPE)
        self.assertEqual(event.source, GRAMMAR_GUARD_SOURCE)
        self.assertEqual(event.text, "an")
        self.assertEqual(event.details["alternate_fragment"], "a")
        self.assertEqual(event.details["evidence_sources"], ["zoom"])
        self.assertFalse(event.reviewed)
        self.assertTrue(turn.manual_review)
        self.assertEqual(project.metrics["grammar_preservation_candidates"], 1)
        self.assertEqual(
            project.metrics["grammar_preservation_candidates_unreviewed"],
            1,
        )

    def test_have_and_preposition_differences_create_neutral_candidates(self) -> None:
        examples = [
            (
                "It has a lamp",
                "It have a lamp",
                "have_form_change",
                "has",
                "have",
            ),
            (
                "I am able to fly today",
                "I am able for fly today",
                "preposition_form_change",
                "to",
                "for",
            ),
        ]

        for final, source, pattern, observed, alternate in examples:
            with self.subTest(pattern=pattern):
                turn = Turn(
                    turn_id=1,
                    speaker="Student",
                    zoom_text=source,
                    final_text=final,
                )
                project = ProjectData(turns=[turn])

                refresh_grammar_preservation_events(project)

                [event] = grammar_events_for_turn(project, turn.turn_id)
                self.assertEqual(turn.final_text, final)
                self.assertEqual(event.details["pattern"], pattern)
                self.assertEqual(event.details["observed_fragment"], observed)
                self.assertEqual(event.details["alternate_fragment"], alternate)
                self.assertEqual(event.details["decision"], "candidate")

    def test_case_only_source_variants_share_one_stable_candidate(self) -> None:
        turn = Turn(
            turn_id=1,
            speaker="Student",
            zoom_text="I saw a enemy today",
            model_text="I saw A enemy today",
            final_text="I saw an enemy today",
        )
        project = ProjectData(turns=[turn])

        refresh_grammar_preservation_events(project)

        [event] = grammar_events_for_turn(project, turn.turn_id)
        self.assertEqual(
            event.details["evidence_sources"],
            ["additional_model", "zoom"],
        )
        self.assertEqual(
            event.details["source_variants"],
            [
                {"source": "additional_model", "alternate_fragment": "A"},
                {"source": "zoom", "alternate_fragment": "a"},
            ],
        )
        set_grammar_events_confirmed([event], True)

        turn.model_text = "I saw a enemy today"
        self.assertEqual(refresh_grammar_preservation_events(project), set())
        [stable] = grammar_events_for_turn(project, turn.turn_id)
        self.assertEqual(stable.event_id, event.event_id)
        self.assertTrue(stable.reviewed)

    def test_identical_sources_keep_the_learner_form_without_a_candidate(self) -> None:
        literal = "I have a idea"
        turn = Turn(
            turn_id=1,
            speaker_raw="Student",
            speaker="Student",
            zoom_text=literal,
            chatgpt_text=literal,
            model_text=literal,
            final_text=literal,
            model_confidence=0.99,
            manual_review=True,
        )
        project = ProjectData(turns=[turn])

        analyze_turns(project, predictor=_AlwaysAcceptablePredictor())

        self.assertEqual(turn.final_text, literal)
        self.assertEqual(grammar_events_for_turn(project, turn.turn_id), [])
        self.assertFalse(turn.manual_review)

    def test_teacher_candidate_does_not_apply_the_learner_review_gate(self) -> None:
        turn = _source_disagreement_turn(speaker="Teacher")
        project = ProjectData(turns=[turn])

        analyze_turns(project, predictor=_AlwaysAcceptablePredictor())

        [event] = grammar_events_for_turn(project, turn.turn_id)
        self.assertFalse(event.reviewed)
        self.assertFalse(turn.manual_review)

    def test_named_teacher_is_not_assumed_to_be_the_learner(self) -> None:
        turn = _source_disagreement_turn(speaker="Alice")
        project = ProjectData(
            metadata=ProjectMetadata(learner_id="Dana"),
            turns=[turn],
        )

        analyze_turns(project, predictor=_AlwaysAcceptablePredictor())

        self.assertEqual(len(grammar_events_for_turn(project, turn.turn_id)), 1)
        self.assertFalse(turn.manual_review)

    def test_project_learner_name_applies_the_review_gate(self) -> None:
        turn = _source_disagreement_turn(speaker="Dana")
        project = ProjectData(
            metadata=ProjectMetadata(learner_id="Dana"),
            turns=[turn],
        )

        analyze_turns(project, predictor=_AlwaysAcceptablePredictor())

        self.assertEqual(len(grammar_events_for_turn(project, turn.turn_id)), 1)
        self.assertTrue(turn.manual_review)

    def test_learner_name_beginning_with_ai_is_not_mistaken_for_ai_role(self) -> None:
        turn = _source_disagreement_turn(speaker="Aiden")
        project = ProjectData(
            metadata=ProjectMetadata(learner_id="Aiden"),
            turns=[turn],
        )

        analyze_turns(project, predictor=_AlwaysAcceptablePredictor())

        self.assertEqual(len(grammar_events_for_turn(project, turn.turn_id)), 1)
        self.assertTrue(turn.manual_review)

    def test_ambiguous_turn_flags_suppress_automatic_grammar_evidence(self) -> None:
        ambiguity_flags = (
            "overlapping_speech",
            "unclear_speech",
            "hebrew_switch",
            "self_correction",
        )

        for flag in ambiguity_flags:
            with self.subTest(flag=flag):
                turn = _source_disagreement_turn()
                setattr(turn, flag, True)
                project = ProjectData(turns=[turn])

                refresh_grammar_preservation_events(project)

                self.assertEqual(
                    grammar_events_for_turn(project, turn.turn_id),
                    [],
                )

    def test_repetition_is_suppressed_but_filler_does_not_hide_local_evidence(self) -> None:
        repeated = Turn(
            turn_id=1,
            speaker="Student",
            zoom_text="I did go home",
            final_text="I did did go home",
        )
        filler = Turn(
            turn_id=2,
            speaker="Student",
            zoom_text="Um I has a lamp",
            final_text="Um I have a lamp",
        )
        project = ProjectData(turns=[repeated, filler])

        refresh_grammar_preservation_events(project)

        self.assertEqual(grammar_events_for_turn(project, repeated.turn_id), [])
        [event] = grammar_events_for_turn(project, filler.turn_id)
        self.assertEqual(event.details["pattern"], "have_form_change")

    def test_confirmed_stable_evidence_survives_refresh_and_model_roundtrip(self) -> None:
        turn = _source_disagreement_turn()
        project = ProjectData(turns=[turn])
        refresh_grammar_preservation_events(project)
        [event] = grammar_events_for_turn(project, turn.turn_id)
        set_grammar_events_confirmed([event], True)
        original_id = event.event_id
        original_details = dict(event.details)

        invalidated = refresh_grammar_preservation_events(project)

        [stable] = grammar_events_for_turn(project, turn.turn_id)
        self.assertEqual(invalidated, set())
        self.assertEqual(stable.event_id, original_id)
        self.assertTrue(stable.reviewed)
        self.assertEqual(stable.details["decision"], "confirmed_as_spoken")
        self.assertEqual(stable.details, original_details)

        restored = ProjectData.from_dict(project.to_dict())
        [restored_event] = grammar_events_for_turn(restored, turn.turn_id)
        self.assertEqual(restored_event.event_id, original_id)
        self.assertTrue(restored_event.reviewed)
        self.assertEqual(restored_event.details, stable.details)

    def test_typography_edit_does_not_invalidate_spoken_word_confirmation(self) -> None:
        turn = _source_disagreement_turn()
        project = ProjectData(turns=[turn])
        refresh_grammar_preservation_events(project)
        [event] = grammar_events_for_turn(project, turn.turn_id)
        set_grammar_events_confirmed([event], True)

        turn.final_text = "  i saw an enemy today!  "
        invalidated = refresh_grammar_preservation_events(project)

        [stable] = grammar_events_for_turn(project, turn.turn_id)
        self.assertEqual(invalidated, set())
        self.assertEqual(stable.event_id, event.event_id)
        self.assertTrue(stable.reviewed)
        self.assertEqual(stable.details["observed_fragment"], "an")

    def test_relevant_final_edit_invalidates_confirmation_and_resets_candidate(self) -> None:
        turn = Turn(
            turn_id=1,
            speaker="Student",
            zoom_text="I saw a enemy today",
            final_text="I saw an enemy today",
        )
        project = ProjectData(turns=[turn])
        refresh_grammar_preservation_events(project)
        [old_event] = grammar_events_for_turn(project, turn.turn_id)
        set_grammar_events_confirmed([old_event], True)

        turn.final_text = "I saw enemy today"
        invalidated = refresh_grammar_preservation_events(project)

        [new_event] = grammar_events_for_turn(project, turn.turn_id)
        self.assertIn(old_event.event_id, invalidated)
        self.assertIn(new_event.event_id, invalidated)
        self.assertNotEqual(new_event.event_id, old_event.event_id)
        self.assertFalse(new_event.reviewed)
        self.assertEqual(new_event.details["decision"], "candidate")
        self.assertEqual(new_event.details["operation"], "insert")
        self.assertEqual(new_event.details["alternate_fragment"], "a")
        self.assertEqual(turn.final_text, "I saw enemy today")


class GrammarEventGuiTests(unittest.TestCase):
    def test_gui_cannot_clear_manual_review_while_candidate_is_unconfirmed(self) -> None:
        turn = _source_disagreement_turn(manual_review=True)
        project = ProjectData(turns=[turn])
        refresh_grammar_preservation_events(project)
        app = _editor_stub(
            project,
            final_text=turn.final_text,
            grammar_confirmed=False,
            manual_review=False,
        )

        TranscriptionApp.save_editor_to_turn(
            app,
            silent=True,
            refresh_table=False,
        )

        [event] = grammar_events_for_turn(project, turn.turn_id)
        self.assertFalse(event.reviewed)
        self.assertEqual(event.details["decision"], "candidate")
        self.assertTrue(turn.manual_review)
        self.assertTrue(app.manual_review_var.get())

    def test_gui_confirmed_as_spoken_can_release_manual_review(self) -> None:
        turn = _source_disagreement_turn(manual_review=True)
        project = ProjectData(turns=[turn])
        refresh_grammar_preservation_events(project)
        app = _editor_stub(
            project,
            final_text=turn.final_text,
            grammar_confirmed=True,
            manual_review=False,
        )

        TranscriptionApp.save_editor_to_turn(
            app,
            silent=True,
            refresh_table=False,
        )

        [event] = grammar_events_for_turn(project, turn.turn_id)
        self.assertTrue(event.reviewed)
        self.assertEqual(event.details["decision"], "confirmed_as_spoken")
        self.assertFalse(turn.manual_review)
        self.assertFalse(app.manual_review_var.get())

    def test_prechecked_control_cannot_confirm_newly_changed_evidence(self) -> None:
        turn = Turn(
            turn_id=1,
            speaker_raw="Student",
            speaker="Student",
            zoom_text="I saw a enemy today",
            final_text="I saw an enemy today",
            manual_review=False,
        )
        project = ProjectData(turns=[turn])
        refresh_grammar_preservation_events(project)
        [old_event] = grammar_events_for_turn(project, turn.turn_id)
        set_grammar_events_confirmed([old_event], True)
        app = _editor_stub(
            project,
            final_text="I saw enemy today",
            grammar_confirmed=True,
            manual_review=False,
        )

        TranscriptionApp.save_editor_to_turn(
            app,
            silent=True,
            refresh_table=False,
        )

        [new_event] = grammar_events_for_turn(project, turn.turn_id)
        self.assertNotEqual(new_event.event_id, old_event.event_id)
        self.assertFalse(new_event.reviewed)
        self.assertEqual(new_event.details["decision"], "candidate")
        self.assertFalse(app.grammar_reviewed_var.get())
        self.assertTrue(turn.manual_review)
        self.assertTrue(app.manual_review_var.get())
        self.assertEqual(turn.final_text, "I saw enemy today")


if __name__ == "__main__":
    unittest.main()
