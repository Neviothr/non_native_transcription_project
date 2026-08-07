from __future__ import annotations

import json
import sys
import unittest
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.grammar_preservation import (
    GrammarEditEvidence,
    review_grammar_edit_preservation,
)


class GrammarPreservationReviewTests(unittest.TestCase):
    def test_paper_style_seems_to_seem_is_located_without_rewriting(self) -> None:
        baseline = "i seems to be affected"
        final = "i seem to be affected"

        review = review_grammar_edit_preservation(baseline, final)

        self.assertEqual(review.baseline_text, baseline)
        self.assertEqual(review.final_text, final)
        self.assertTrue(review.requires_review)
        self.assertFalse(review.exact_text_preserved)
        self.assertEqual(len(review.evidence), 1)
        item = review.evidence[0]
        self.assertEqual(item.operation, "replace")
        self.assertEqual(item.pattern, "s_inflection_change")
        self.assertEqual(item.baseline_fragment, "seems")
        self.assertEqual(item.final_fragment, "seem")
        self.assertEqual(
            (item.baseline_token_start, item.baseline_token_end),
            (1, 2),
        )
        self.assertEqual((item.final_token_start, item.final_token_end), (1, 2))
        self.assertEqual(
            baseline[item.baseline_char_start:item.baseline_char_end],
            "seems",
        )
        self.assertEqual(
            final[item.final_char_start:item.final_char_end],
            "seem",
        )
        self.assertEqual((item.left_anchor, item.right_anchor), ("i", "to"))

    def test_form_article_preposition_and_pronoun_patterns_are_neutral_evidence(self) -> None:
        examples = [
            ("It have a lamp", "It has a lamp", "have_form_change"),
            ("I saw a enemy", "I saw an enemy", "article_form_change"),
            ("I am able for fly", "I am able to fly", "preposition_form_change"),
            ("Tell me you name", "Tell me your name", "pronoun_form_change"),
        ]

        for baseline, final, pattern in examples:
            with self.subTest(pattern=pattern):
                review = review_grammar_edit_preservation(baseline, final)
                self.assertEqual([item.pattern for item in review.evidence], [pattern])
                self.assertIn("review", review.evidence[0].rationale.casefold())

    def test_function_word_presence_requires_two_stable_anchors(self) -> None:
        inserted = review_grammar_edit_preservation(
            "I want go now",
            "I want to go now",
        )
        deleted = review_grammar_edit_preservation(
            "I did went home",
            "I went home",
        )

        self.assertEqual(len(inserted.evidence), 1)
        insertion = inserted.evidence[0]
        self.assertEqual(insertion.operation, "insert")
        self.assertEqual(insertion.pattern, "preposition_presence_change")
        self.assertEqual(insertion.baseline_fragment, "")
        self.assertEqual(insertion.final_fragment, "to")
        self.assertEqual((insertion.left_anchor, insertion.right_anchor), ("want", "go"))
        self.assertEqual(
            insertion.baseline_char_start,
            insertion.baseline_char_end,
        )

        self.assertEqual(len(deleted.evidence), 1)
        deletion = deleted.evidence[0]
        self.assertEqual(deletion.operation, "delete")
        self.assertEqual(deletion.pattern, "auxiliary_presence_change")
        self.assertEqual(deletion.baseline_fragment, "did")
        self.assertEqual(deletion.final_fragment, "")
        self.assertEqual((deletion.left_anchor, deletion.right_anchor), ("I", "went"))

        self.assertFalse(
            review_grammar_edit_preservation("go home", "to go home").evidence
        )
        self.assertFalse(
            review_grammar_edit_preservation("I go", "I go to").evidence
        )

    def test_repeated_tokens_do_not_move_the_reported_edit(self) -> None:
        baseline = "I have a have a car"
        final = "I has a have a car"

        first = review_grammar_edit_preservation(baseline, final)
        second = review_grammar_edit_preservation(baseline, final)

        self.assertEqual(first, second)
        self.assertEqual(len(first.evidence), 1)
        item = first.evidence[0]
        self.assertEqual(item.baseline_token_start, 1)
        self.assertEqual(item.final_token_start, 1)
        self.assertEqual(item.baseline_fragment, "have")
        self.assertEqual(item.final_fragment, "has")
        self.assertEqual((item.left_anchor, item.right_anchor), ("I", "a"))

    def test_spelling_content_punctuation_and_large_rewrites_are_not_flagged(self) -> None:
        examples = [
            ("My favorite color is blue.", "My favourite colour is blue."),
            ("The cat sleeps here", "The dog sleeps here"),
            ("The new report arrived", "The news report arrived"),
            ("I seems, okay?", "i seems okay!"),
            ("What do you favourite food?", "What's your favorite food?"),
            ("have", "has"),
        ]

        for baseline, final in examples:
            with self.subTest(baseline=baseline, final=final):
                review = review_grammar_edit_preservation(baseline, final)
                self.assertFalse(review.requires_review)
                self.assertEqual(review.evidence, ())

    def test_contraction_irregular_verb_and_adjacent_order_are_review_evidence(self) -> None:
        examples = [
            (
                "I do not know now",
                "I don't know now",
                ["contraction_form_change"],
            ),
            (
                "I cannot go now",
                "I can't go now",
                ["contraction_form_change"],
            ),
            (
                "I go yesterday",
                "I went yesterday",
                ["irregular_verb_form_change"],
            ),
            (
                "What you are doing",
                "What are you doing",
                ["word_order_change"],
            ),
            (
                "What do you favorite food",
                "What is your favorite food",
                ["auxiliary_form_change", "pronoun_form_change"],
            ),
        ]

        for baseline, final, expected_patterns in examples:
            with self.subTest(baseline=baseline, final=final):
                review = review_grammar_edit_preservation(baseline, final)
                self.assertEqual(
                    [item.pattern for item in review.evidence],
                    expected_patterns,
                )
                self.assertEqual(review.baseline_text, baseline)
                self.assertEqual(review.final_text, final)

    def test_modal_copula_short_inflection_and_reflexive_patterns(self) -> None:
        examples = [
            ("I can go now", "I could go now", "modal_auxiliary_form_change"),
            ("She being ready", "She is ready", "be_form_change"),
            ("Bird fly home", "Bird flies home", "s_inflection_change"),
            ("I use it", "I uses it", "s_inflection_change"),
            ("I hurt me today", "I hurt myself today", "pronoun_form_change"),
            ("I drink yesterday", "I drank yesterday", "irregular_verb_form_change"),
        ]

        for baseline, final, expected_pattern in examples:
            with self.subTest(baseline=baseline, final=final):
                review = review_grammar_edit_preservation(baseline, final)
                self.assertEqual(
                    [item.pattern for item in review.evidence],
                    [expected_pattern],
                )

        inserted_modal = review_grammar_edit_preservation(
            "I go home now",
            "I can go home now",
        )
        self.assertEqual(
            [item.pattern for item in inserted_modal.evidence],
            ["auxiliary_presence_change"],
        )

    def test_exact_text_is_preserved_and_empty_comparisons_are_safe(self) -> None:
        exact = "Um, I have a idea."
        review = review_grammar_edit_preservation(exact, exact)
        empty = review_grammar_edit_preservation("", "")

        self.assertTrue(review.exact_text_preserved)
        self.assertEqual(review.baseline_text, exact)
        self.assertEqual(review.evidence, ())
        self.assertTrue(empty.exact_text_preserved)
        self.assertEqual(empty.evidence, ())

    def test_evidence_is_frozen_and_deterministically_serializable(self) -> None:
        review = review_grammar_edit_preservation(
            "She walk to school",
            "She walks to school",
        )
        self.assertEqual(len(review.evidence), 1)

        first = asdict(review)
        second = asdict(
            review_grammar_edit_preservation(
                "She walk to school",
                "She walks to school",
            )
        )
        self.assertEqual(first, second)
        self.assertEqual(
            json.dumps(first, ensure_ascii=False, sort_keys=True),
            json.dumps(second, ensure_ascii=False, sort_keys=True),
        )
        with self.assertRaises(FrozenInstanceError):
            review.evidence[0].pattern = "changed"  # type: ignore[misc]
        self.assertIsInstance(review.evidence[0], GrammarEditEvidence)

    def test_non_string_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "must both be strings"):
            review_grammar_edit_preservation(None, "text")  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "must both be strings"):
            review_grammar_edit_preservation("text", 4)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
