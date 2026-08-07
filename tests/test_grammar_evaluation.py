from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.grammar_evaluation import (
    GrammarPreservationEvaluation,
    aggregate_grammar_preservation,
    evaluate_grammar_preservation,
)


class GrammarPreservationEvaluationTests(unittest.TestCase):
    def test_one_of_three_annotated_tokens_is_preserved(self) -> None:
        evaluation = evaluate_grammar_preservation(
            "I have@! a@! book for@! school",
            "I has book for school",
        )

        self.assertEqual(evaluation.grammar_error_tokens_evaluated, 3)
        self.assertEqual(evaluation.grammar_error_tokens_preserved, 1)
        self.assertEqual(evaluation.grammar_error_token_substitutions, 1)
        self.assertEqual(evaluation.grammar_error_token_deletions, 1)
        self.assertAlmostEqual(evaluation.grammar_error_preservation_rate, 1 / 3)
        self.assertAlmostEqual(evaluation.unwanted_grammar_correction_rate, 2 / 3)
        self.assertAlmostEqual(evaluation.grammar_error_token_loss_rate, 2 / 3)

    def test_exact_annotated_surfaces_are_fully_preserved(self) -> None:
        evaluation = evaluate_grammar_preservation(
            "She have@! two idea@!.",
            "She have two idea.",
        )

        self.assertEqual(
            evaluation.to_metrics(),
            {
                "grammar_error_tokens_evaluated": 2,
                "grammar_error_tokens_preserved": 2,
                "grammar_error_token_substitutions": 0,
                "grammar_error_token_deletions": 0,
                "grammar_error_preservation_rate": 1.0,
                "unwanted_grammar_correction_rate": 0.0,
                "grammar_error_token_loss_rate": 0.0,
            },
        )

    def test_unannotated_gold_never_infers_grammar_errors(self) -> None:
        evaluation = evaluate_grammar_preservation(
            "She have two idea",
            "She has two ideas",
        )

        self.assertEqual(evaluation.grammar_error_tokens_evaluated, 0)
        self.assertIsNone(evaluation.grammar_error_preservation_rate)
        self.assertIsNone(evaluation.unwanted_grammar_correction_rate)
        self.assertIsNone(evaluation.grammar_error_token_loss_rate)

    def test_insertions_do_not_damage_an_aligned_annotated_token(self) -> None:
        evaluation = evaluate_grammar_preservation(
            "It have@! a plan",
            "Well It really have a plan now",
        )

        self.assertEqual(evaluation.grammar_error_tokens_preserved, 1)
        self.assertEqual(evaluation.grammar_error_token_substitutions, 0)
        self.assertEqual(evaluation.grammar_error_token_deletions, 0)

    def test_straight_and_curly_contractions_are_single_exact_tokens(self) -> None:
        preserved = evaluate_grammar_preservation(
            "It’s@! fine but I don't@! know",
            "It’s really fine but I don't know",
        )
        typography_variant = evaluate_grammar_preservation(
            "It’s@! fine",
            "It's fine",
        )

        self.assertEqual(preserved.grammar_error_tokens_evaluated, 2)
        self.assertEqual(preserved.grammar_error_tokens_preserved, 2)
        self.assertEqual(typography_variant.grammar_error_tokens_preserved, 1)
        self.assertEqual(typography_variant.grammar_error_token_substitutions, 0)

    def test_repeated_surface_is_evaluated_at_its_aligned_occurrence(self) -> None:
        first_occurrence = evaluate_grammar_preservation(
            "I have@! a have a plan",
            "I has a have a plan",
        )
        second_occurrence = evaluate_grammar_preservation(
            "I have a have@! plan",
            "I has a have plan",
        )

        self.assertEqual(first_occurrence.grammar_error_token_substitutions, 1)
        self.assertEqual(first_occurrence.grammar_error_tokens_preserved, 0)
        self.assertEqual(second_occurrence.grammar_error_tokens_preserved, 1)

    def test_per_turn_results_aggregate_counts_before_computing_rates(self) -> None:
        combined = aggregate_grammar_preservation(
            [
                evaluate_grammar_preservation("I have@!", "I have"),
                evaluate_grammar_preservation("two idea@!", "two ideas"),
                evaluate_grammar_preservation("unannotated", "changed"),
            ]
        )

        self.assertIsInstance(combined, GrammarPreservationEvaluation)
        self.assertEqual(combined.grammar_error_tokens_evaluated, 2)
        self.assertEqual(combined.grammar_error_tokens_preserved, 1)
        self.assertEqual(combined.grammar_error_token_substitutions, 1)
        self.assertEqual(combined.grammar_error_preservation_rate, 0.5)

    def test_non_string_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "must both be strings"):
            evaluate_grammar_preservation(None, "text")  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "must both be strings"):
            evaluate_grammar_preservation("text", 4)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
