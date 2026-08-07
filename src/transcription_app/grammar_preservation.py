"""Conservative evidence for possible learner-grammar normalization.

This module compares two immutable strings: a baseline transcript intended to
represent what the learner said and a later final transcript.  It does not
judge either string's grammaticality and never rewrites either input.  Instead,
it reports only small, location-supported edits whose surface form resembles a
common grammar-edit pattern.  The evidence is intended for human review, not
automatic correction.

The implementation deliberately favors precision over recall.  It ignores
spelling and content-word replacements, punctuation/case changes, and large
replacement blocks. It includes only explicitly enumerated contraction,
irregular-verb, and adjacent word-order patterns.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher


_TOKEN_RE = re.compile(
    r"[^\W_]+(?:['\u2019][^\W_]+)*|\d+(?:[.,]\d+)*",
    re.UNICODE,
)

_ARTICLE_FORMS = frozenset({"a", "an", "the"})
_PREPOSITION_FORMS = frozenset(
    {
        "about",
        "after",
        "at",
        "before",
        "between",
        "by",
        "during",
        "for",
        "from",
        "in",
        "into",
        "of",
        "off",
        "on",
        "onto",
        "over",
        "through",
        "to",
        "under",
        "with",
        "without",
    }
)
_FORM_FAMILIES: tuple[tuple[str, frozenset[str]], ...] = (
    ("be_form_change", frozenset({"am", "is", "are", "was", "were", "be", "been", "being"})),
    ("have_form_change", frozenset({"have", "has", "had"})),
    ("do_form_change", frozenset({"do", "does", "did"})),
    ("modal_auxiliary_form_change", frozenset({"can", "could"})),
    ("modal_auxiliary_form_change", frozenset({"will", "would"})),
    ("modal_auxiliary_form_change", frozenset({"shall", "should"})),
    ("modal_auxiliary_form_change", frozenset({"may", "might"})),
    ("negative_be_form_change", frozenset({"isn't", "aren't", "wasn't", "weren't"})),
    ("negative_have_form_change", frozenset({"haven't", "hasn't", "hadn't"})),
    ("negative_do_form_change", frozenset({"don't", "doesn't", "didn't"})),
    ("pronoun_form_change", frozenset({"i", "me", "my", "mine", "myself"})),
    ("pronoun_form_change", frozenset({"we", "us", "our", "ours", "ourselves"})),
    ("pronoun_form_change", frozenset({"you", "your", "yours", "yourself", "yourselves"})),
    ("pronoun_form_change", frozenset({"he", "him", "his", "himself"})),
    ("pronoun_form_change", frozenset({"she", "her", "hers", "herself"})),
    ("pronoun_form_change", frozenset({"it", "its", "itself"})),
    ("pronoun_form_change", frozenset({"they", "them", "their", "theirs", "themselves"})),
    ("demonstrative_form_change", frozenset({"this", "these"})),
    ("demonstrative_form_change", frozenset({"that", "those"})),
)

_AUXILIARY_PATTERNS = {
    "be_form_change",
    "have_form_change",
    "do_form_change",
    "modal_auxiliary_form_change",
    "negative_be_form_change",
    "negative_have_form_change",
    "negative_do_form_change",
}
_AUXILIARY_FORMS = frozenset().union(
    *(forms for pattern, forms in _FORM_FAMILIES if pattern in _AUXILIARY_PATTERNS)
)
_PRONOUN_FORMS = frozenset().union(
    *(forms for pattern, forms in _FORM_FAMILIES if pattern == "pronoun_form_change")
)
_IRREGULAR_VERB_FAMILIES: tuple[frozenset[str], ...] = (
    frozenset({"go", "goes", "went", "gone"}),
    frozenset({"come", "comes", "came"}),
    frozenset({"see", "sees", "saw", "seen"}),
    frozenset({"take", "takes", "took", "taken"}),
    frozenset({"make", "makes", "made"}),
    frozenset({"get", "gets", "got", "gotten"}),
    frozenset({"say", "says", "said"}),
    frozenset({"tell", "tells", "told"}),
    frozenset({"think", "thinks", "thought"}),
    frozenset({"know", "knows", "knew", "known"}),
    frozenset({"eat", "eats", "ate", "eaten"}),
    frozenset({"write", "writes", "wrote", "written"}),
    frozenset({"speak", "speaks", "spoke", "spoken"}),
    frozenset({"give", "gives", "gave", "given"}),
    frozenset({"find", "finds", "found"}),
    frozenset({"buy", "buys", "bought"}),
    frozenset({"bring", "brings", "brought"}),
    frozenset({"teach", "teaches", "taught"}),
    frozenset({"catch", "catches", "caught"}),
    frozenset({"feel", "feels", "felt"}),
    frozenset({"leave", "leaves", "left"}),
    frozenset({"meet", "meets", "met"}),
    frozenset({"run", "runs", "ran"}),
    frozenset({"sit", "sits", "sat"}),
    frozenset({"stand", "stands", "stood"}),
    frozenset({"understand", "understands", "understood"}),
    frozenset({"drink", "drinks", "drank", "drunk"}),
    frozenset({"begin", "begins", "began", "begun"}),
    frozenset({"sing", "sings", "sang", "sung"}),
    frozenset({"swim", "swims", "swam", "swum"}),
    frozenset({"drive", "drives", "drove", "driven"}),
    frozenset({"choose", "chooses", "chose", "chosen"}),
    frozenset({"break", "breaks", "broke", "broken"}),
    frozenset({"fall", "falls", "fell", "fallen"}),
    frozenset({"forget", "forgets", "forgot", "forgotten"}),
    frozenset({"keep", "keeps", "kept"}),
    frozenset({"sleep", "sleeps", "slept"}),
    frozenset({"build", "builds", "built"}),
    frozenset({"send", "sends", "sent"}),
    frozenset({"spend", "spends", "spent"}),
    frozenset({"lose", "loses", "lost"}),
    frozenset({"win", "wins", "won"}),
    frozenset({"hold", "holds", "held"}),
    frozenset({"hear", "hears", "heard"}),
    frozenset({"pay", "pays", "paid"}),
    frozenset({"sell", "sells", "sold"}),
    frozenset({"wear", "wears", "wore", "worn"}),
)
_CONTRACTION_EXPANSIONS: dict[str, tuple[tuple[str, ...], ...]] = {
    "cannot": (("can", "not"),),
    "can't": (("can", "not"),),
    "couldn't": (("could", "not"),),
    "didn't": (("did", "not"),),
    "doesn't": (("does", "not"),),
    "don't": (("do", "not"),),
    "hadn't": (("had", "not"),),
    "hasn't": (("has", "not"),),
    "haven't": (("have", "not"),),
    "isn't": (("is", "not"),),
    "aren't": (("are", "not"),),
    "wasn't": (("was", "not"),),
    "weren't": (("were", "not"),),
    "mustn't": (("must", "not"),),
    "shouldn't": (("should", "not"),),
    "won't": (("will", "not"),),
    "wouldn't": (("would", "not"),),
    "i'm": (("i", "am"),),
    "you're": (("you", "are"),),
    "we're": (("we", "are"),),
    "they're": (("they", "are"),),
    "he's": (("he", "is"), ("he", "has")),
    "she's": (("she", "is"), ("she", "has")),
    "it's": (("it", "is"), ("it", "has")),
    "i've": (("i", "have"),),
    "you've": (("you", "have"),),
    "we've": (("we", "have"),),
    "they've": (("they", "have"),),
    "i'll": (("i", "will"),),
    "you'll": (("you", "will"),),
    "he'll": (("he", "will"),),
    "she'll": (("she", "will"),),
    "we'll": (("we", "will"),),
    "they'll": (("they", "will"),),
    "i'd": (("i", "would"), ("i", "had")),
    "you'd": (("you", "would"), ("you", "had")),
    "he'd": (("he", "would"), ("he", "had")),
    "she'd": (("she", "would"), ("she", "had")),
    "we'd": (("we", "would"), ("we", "had")),
    "they'd": (("they", "would"), ("they", "had")),
}
_INFLECTION_FALSE_FRIENDS = {
    frozenset({"new", "news"}),
}
_INSERT_DELETE_PATTERNS = {
    **{form: "article_presence_change" for form in _ARTICLE_FORMS},
    **{form: "preposition_presence_change" for form in _PREPOSITION_FORMS},
    **{form: "auxiliary_presence_change" for form in _AUXILIARY_FORMS},
    **{form: "pronoun_presence_change" for form in _PRONOUN_FORMS},
    "not": "negation_presence_change",
}

_RATIONALES = {
    "article_form_change": (
        "An aligned a/an form changed; review whether the learner's surface "
        "form was normalized."
    ),
    "preposition_form_change": (
        "One aligned preposition was replaced by another; review the audio "
        "before accepting the change."
    ),
    "be_form_change": (
        "Aligned forms from the be paradigm differ; this is review evidence, "
        "not a grammaticality judgment."
    ),
    "have_form_change": (
        "Aligned forms from the have paradigm differ; review whether the "
        "baseline learner form should remain verbatim."
    ),
    "do_form_change": (
        "Aligned forms from the do paradigm differ; review whether the "
        "baseline learner form should remain verbatim."
    ),
    "modal_auxiliary_form_change": (
        "Aligned forms from one modal auxiliary family differ; review whether "
        "a source normalized the learner's surface form."
    ),
    "negative_be_form_change": (
        "Aligned negative be forms differ; review the original audio and "
        "retain the learner's audible form."
    ),
    "negative_have_form_change": (
        "Aligned negative have forms differ; review the original audio and "
        "retain the learner's audible form."
    ),
    "negative_do_form_change": (
        "Aligned negative do forms differ; review the original audio and "
        "retain the learner's audible form."
    ),
    "pronoun_form_change": (
        "Aligned forms from one pronoun paradigm differ; review whether this "
        "was an automatic normalization."
    ),
    "demonstrative_form_change": (
        "Aligned demonstrative forms differ; review whether this was an "
        "automatic normalization."
    ),
    "s_inflection_change": (
        "The aligned words differ only by a conservative -s/-es/ies surface "
        "pattern; review rather than assuming a correction."
    ),
    "ed_inflection_change": (
        "The aligned words differ only by a conservative -ed surface pattern; "
        "review rather than assuming a correction."
    ),
    "ing_inflection_change": (
        "The aligned words differ only by a conservative -ing surface pattern; "
        "review rather than assuming a correction."
    ),
    "article_presence_change": (
        "A single article was inserted or deleted between two stable aligned "
        "anchors; review the baseline audio."
    ),
    "preposition_presence_change": (
        "A single preposition was inserted or deleted between two stable "
        "aligned anchors; review the baseline audio."
    ),
    "auxiliary_presence_change": (
        "A single auxiliary/paradigm form was inserted or deleted between two "
        "stable aligned anchors; review the baseline audio."
    ),
    "negation_presence_change": (
        "The word 'not' was inserted or deleted between two stable aligned "
        "anchors; review the baseline audio."
    ),
    "pronoun_presence_change": (
        "A single pronoun form was inserted or deleted between two stable "
        "anchors; review the learner's audible wording."
    ),
    "auxiliary_form_change": (
        "Aligned auxiliary forms differ; review whether one source normalized "
        "the learner's surface construction."
    ),
    "irregular_verb_form_change": (
        "Aligned forms of one enumerated irregular verb differ; review the "
        "audio rather than assuming a tense correction."
    ),
    "contraction_form_change": (
        "A contraction and its enumerated expanded form differ between aligned "
        "sources; retain the form actually spoken."
    ),
    "word_order_change": (
        "Two adjacent aligned words appear in reverse order; review whether a "
        "source normalized the learner's spoken order."
    ),
}


@dataclass(frozen=True, slots=True)
class GrammarEditEvidence:
    """One neutral, location-supported reason to review a text edit.

    Character and token ranges are half-open.  Empty ranges represent an
    insertion or deletion boundary.  Every field is JSON-compatible so
    ``dataclasses.asdict`` produces deterministic serializable evidence.
    """

    operation: str
    pattern: str
    baseline_fragment: str
    final_fragment: str
    baseline_char_start: int
    baseline_char_end: int
    final_char_start: int
    final_char_end: int
    baseline_token_start: int
    baseline_token_end: int
    final_token_start: int
    final_token_end: int
    left_anchor: str
    right_anchor: str
    rationale: str


@dataclass(frozen=True, slots=True)
class GrammarPreservationReview:
    """Immutable comparison result retaining both exact source strings."""

    baseline_text: str
    final_text: str
    evidence: tuple[GrammarEditEvidence, ...]

    @property
    def requires_review(self) -> bool:
        return bool(self.evidence)

    @property
    def exact_text_preserved(self) -> bool:
        return self.baseline_text == self.final_text


@dataclass(frozen=True, slots=True)
class _Token:
    raw: str
    normalized: str
    char_start: int
    char_end: int


def _normalize_token(value: str) -> str:
    return (
        unicodedata.normalize("NFKC", value)
        .replace("\u2019", "'")
        .casefold()
    )


def _tokens(text: str) -> tuple[_Token, ...]:
    return tuple(
        _Token(
            raw=match.group(0),
            normalized=_normalize_token(match.group(0)),
            char_start=match.start(),
            char_end=match.end(),
        )
        for match in _TOKEN_RE.finditer(text)
    )


def _stable_anchors(
    baseline: tuple[_Token, ...],
    final: tuple[_Token, ...],
    baseline_start: int,
    baseline_end: int,
    final_start: int,
    final_end: int,
) -> tuple[str, str]:
    left = ""
    if baseline_start > 0 and final_start > 0:
        baseline_left = baseline[baseline_start - 1]
        final_left = final[final_start - 1]
        if baseline_left.normalized == final_left.normalized:
            left = baseline_left.raw

    right = ""
    if baseline_end < len(baseline) and final_end < len(final):
        baseline_right = baseline[baseline_end]
        final_right = final[final_end]
        if baseline_right.normalized == final_right.normalized:
            right = baseline_right.raw
    return left, right


def _span(
    text: str,
    tokens: tuple[_Token, ...],
    start: int,
    end: int,
) -> tuple[int, int, str]:
    if start < end:
        char_start = tokens[start].char_start
        char_end = tokens[end - 1].char_end
        return char_start, char_end, text[char_start:char_end]
    boundary = tokens[start].char_start if start < len(tokens) else len(text)
    return boundary, boundary, ""


def _form_family_pattern(baseline: str, final: str) -> str | None:
    for pattern, forms in _FORM_FAMILIES:
        if baseline in forms and final in forms:
            return pattern
    return None


def _inflection_candidates(word: str) -> dict[str, frozenset[str]]:
    """Return deliberately narrow possible roots and surface suffix classes."""

    if not word.isascii() or not word.isalpha() or len(word) < 3:
        return {word: frozenset({"base"})}

    candidates: dict[str, set[str]] = {word: {"base"}}

    def add(root: str, feature: str, *, minimum_root_length: int = 3) -> None:
        if len(root) >= minimum_root_length:
            candidates.setdefault(root, set()).add(feature)

    if word.endswith("ies") and len(word) > 4:
        add(word[:-3] + "y", "s", minimum_root_length=3)
    if word.endswith("es") and len(word) > 4:
        add(word[:-2], "s", minimum_root_length=4)
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        add(word[:-1], "s", minimum_root_length=3)

    if word.endswith("ied") and len(word) > 4:
        add(word[:-3] + "y", "ed")
    if word.endswith("ed") and len(word) > 4:
        stem = word[:-2]
        add(stem, "ed")
        add(word[:-1], "ed")  # loved -> love
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            add(stem[:-1], "ed")  # stopped -> stop

    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        add(stem, "ing")
        add(stem + "e", "ing")  # making -> make
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            add(stem[:-1], "ing")  # running -> run

    return {root: frozenset(features) for root, features in candidates.items()}


def _inflection_pattern(baseline: str, final: str) -> str | None:
    if frozenset({baseline, final}) in _INFLECTION_FALSE_FRIENDS:
        return None
    baseline_candidates = _inflection_candidates(baseline)
    final_candidates = _inflection_candidates(final)
    common_roots = sorted(set(baseline_candidates) & set(final_candidates))
    for root in common_roots:
        baseline_features = baseline_candidates[root]
        final_features = final_candidates[root]
        if baseline_features == final_features:
            continue
        combined = baseline_features | final_features
        if "s" in combined:
            return "s_inflection_change"
        if "ed" in combined:
            return "ed_inflection_change"
        if "ing" in combined:
            return "ing_inflection_change"
    return None


def _replacement_pattern(baseline: str, final: str) -> str | None:
    if baseline == final:
        return None
    if baseline in _ARTICLE_FORMS and final in _ARTICLE_FORMS:
        return "article_form_change"
    family_pattern = _form_family_pattern(baseline, final)
    if family_pattern is not None:
        return family_pattern
    if baseline in _AUXILIARY_FORMS and final in _AUXILIARY_FORMS:
        return "auxiliary_form_change"
    if baseline in _PREPOSITION_FORMS and final in _PREPOSITION_FORMS:
        return "preposition_form_change"
    if any(
        baseline in family and final in family
        for family in _IRREGULAR_VERB_FAMILIES
    ):
        return "irregular_verb_form_change"
    return _inflection_pattern(baseline, final)


def _contraction_equivalent(
    baseline: tuple[str, ...],
    final: tuple[str, ...],
) -> bool:
    def expansions(values: tuple[str, ...]) -> set[tuple[str, ...]]:
        if len(values) == 1 and values[0] in _CONTRACTION_EXPANSIONS:
            return set(_CONTRACTION_EXPANSIONS[values[0]])
        return {values}

    return baseline != final and bool(expansions(baseline) & expansions(final))


def _adjacent_swap_index(
    baseline: list[str],
    final: list[str],
) -> int | None:
    if len(baseline) != len(final) or len(baseline) < 3:
        return None
    differences = [
        index
        for index, (left, right) in enumerate(zip(baseline, final))
        if left != right
    ]
    if len(differences) != 2 or differences[1] != differences[0] + 1:
        return None
    index = differences[0]
    if (
        baseline[index] == final[index + 1]
        and baseline[index + 1] == final[index]
    ):
        return index
    return None


def _evidence(
    *,
    baseline_text: str,
    final_text: str,
    baseline_tokens: tuple[_Token, ...],
    final_tokens: tuple[_Token, ...],
    baseline_start: int,
    baseline_end: int,
    final_start: int,
    final_end: int,
    operation: str,
    pattern: str,
    left_anchor: str,
    right_anchor: str,
) -> GrammarEditEvidence:
    baseline_char_start, baseline_char_end, baseline_fragment = _span(
        baseline_text,
        baseline_tokens,
        baseline_start,
        baseline_end,
    )
    final_char_start, final_char_end, final_fragment = _span(
        final_text,
        final_tokens,
        final_start,
        final_end,
    )
    return GrammarEditEvidence(
        operation=operation,
        pattern=pattern,
        baseline_fragment=baseline_fragment,
        final_fragment=final_fragment,
        baseline_char_start=baseline_char_start,
        baseline_char_end=baseline_char_end,
        final_char_start=final_char_start,
        final_char_end=final_char_end,
        baseline_token_start=baseline_start,
        baseline_token_end=baseline_end,
        final_token_start=final_start,
        final_token_end=final_end,
        left_anchor=left_anchor,
        right_anchor=right_anchor,
        rationale=_RATIONALES[pattern],
    )


def review_grammar_edit_preservation(
    baseline_text: str,
    final_text: str,
) -> GrammarPreservationReview:
    """Locate conservative evidence of possible grammar-like normalization.

    Replacements must match a closed form family, enumerated irregular verb,
    narrow English inflection, contraction expansion, or one adjacent swap and
    have an exact adjacent anchor. A one-token insertion/deletion is reported
    only for selected function words and only when exact anchors exist on both
    sides. Other edits remain deliberately unclassified.
    """

    if not isinstance(baseline_text, str) or not isinstance(final_text, str):
        raise TypeError("baseline_text and final_text must both be strings")

    baseline_tokens = _tokens(baseline_text)
    final_tokens = _tokens(final_text)
    baseline_values = [token.normalized for token in baseline_tokens]
    final_values = [token.normalized for token in final_tokens]

    swap_index = _adjacent_swap_index(baseline_values, final_values)
    if swap_index is not None:
        left_anchor, right_anchor = _stable_anchors(
            baseline_tokens,
            final_tokens,
            swap_index,
            swap_index + 2,
            swap_index,
            swap_index + 2,
        )
        if left_anchor or right_anchor:
            item = _evidence(
                baseline_text=baseline_text,
                final_text=final_text,
                baseline_tokens=baseline_tokens,
                final_tokens=final_tokens,
                baseline_start=swap_index,
                baseline_end=swap_index + 2,
                final_start=swap_index,
                final_end=swap_index + 2,
                operation="reorder",
                pattern="word_order_change",
                left_anchor=left_anchor,
                right_anchor=right_anchor,
            )
            return GrammarPreservationReview(
                baseline_text=baseline_text,
                final_text=final_text,
                evidence=(item,),
            )
    matcher = SequenceMatcher(
        None,
        baseline_values,
        final_values,
        autojunk=False,
    )

    evidence: list[GrammarEditEvidence] = []
    for tag, baseline_start, baseline_end, final_start, final_end in matcher.get_opcodes():
        if tag == "equal":
            continue

        baseline_length = baseline_end - baseline_start
        final_length = final_end - final_start
        left_anchor, right_anchor = _stable_anchors(
            baseline_tokens,
            final_tokens,
            baseline_start,
            baseline_end,
            final_start,
            final_end,
        )

        if tag == "replace" and baseline_length == final_length == 1:
            if not (left_anchor or right_anchor):
                continue
            pattern = _replacement_pattern(
                baseline_tokens[baseline_start].normalized,
                final_tokens[final_start].normalized,
            )
            if pattern is not None:
                evidence.append(
                    _evidence(
                        baseline_text=baseline_text,
                        final_text=final_text,
                        baseline_tokens=baseline_tokens,
                        final_tokens=final_tokens,
                        baseline_start=baseline_start,
                        baseline_end=baseline_end,
                        final_start=final_start,
                        final_end=final_end,
                        operation="replace",
                        pattern=pattern,
                        left_anchor=left_anchor,
                        right_anchor=right_anchor,
                    )
                )
                continue

        if tag == "replace" and (left_anchor or right_anchor):
            baseline_block = tuple(
                token.normalized
                for token in baseline_tokens[baseline_start:baseline_end]
            )
            final_block = tuple(
                token.normalized
                for token in final_tokens[final_start:final_end]
            )
            if _contraction_equivalent(baseline_block, final_block):
                evidence.append(
                    _evidence(
                        baseline_text=baseline_text,
                        final_text=final_text,
                        baseline_tokens=baseline_tokens,
                        final_tokens=final_tokens,
                        baseline_start=baseline_start,
                        baseline_end=baseline_end,
                        final_start=final_start,
                        final_end=final_end,
                        operation="replace",
                        pattern="contraction_form_change",
                        left_anchor=left_anchor,
                        right_anchor=right_anchor,
                    )
                )
                continue

            if 1 < baseline_length == final_length <= 3:
                patterns = [
                    _replacement_pattern(left, right)
                    for left, right in zip(baseline_block, final_block)
                ]
                if all(pattern is not None for pattern in patterns):
                    for offset, pattern in enumerate(patterns):
                        evidence.append(
                            _evidence(
                                baseline_text=baseline_text,
                                final_text=final_text,
                                baseline_tokens=baseline_tokens,
                                final_tokens=final_tokens,
                                baseline_start=baseline_start + offset,
                                baseline_end=baseline_start + offset + 1,
                                final_start=final_start + offset,
                                final_end=final_start + offset + 1,
                                operation="replace",
                                pattern=str(pattern),
                                left_anchor=left_anchor,
                                right_anchor=right_anchor,
                            )
                        )
                    continue

        if tag == "insert" and baseline_length == 0 and final_length == 1:
            changed_word = final_tokens[final_start].normalized
            pattern = _INSERT_DELETE_PATTERNS.get(changed_word)
            if pattern is None or not (left_anchor and right_anchor):
                continue
            evidence.append(
                _evidence(
                    baseline_text=baseline_text,
                    final_text=final_text,
                    baseline_tokens=baseline_tokens,
                    final_tokens=final_tokens,
                    baseline_start=baseline_start,
                    baseline_end=baseline_end,
                    final_start=final_start,
                    final_end=final_end,
                    operation="insert",
                    pattern=pattern,
                    left_anchor=left_anchor,
                    right_anchor=right_anchor,
                )
            )
            continue

        if tag == "delete" and baseline_length == 1 and final_length == 0:
            changed_word = baseline_tokens[baseline_start].normalized
            pattern = _INSERT_DELETE_PATTERNS.get(changed_word)
            if pattern is None or not (left_anchor and right_anchor):
                continue
            evidence.append(
                _evidence(
                    baseline_text=baseline_text,
                    final_text=final_text,
                    baseline_tokens=baseline_tokens,
                    final_tokens=final_tokens,
                    baseline_start=baseline_start,
                    baseline_end=baseline_end,
                    final_start=final_start,
                    final_end=final_end,
                    operation="delete",
                    pattern=pattern,
                    left_anchor=left_anchor,
                    right_anchor=right_anchor,
                )
            )

    ordered = tuple(
        sorted(
            evidence,
            key=lambda item: (
                item.baseline_token_start,
                item.final_token_start,
                item.operation,
                item.pattern,
                item.baseline_fragment,
                item.final_fragment,
            ),
        )
    )
    return GrammarPreservationReview(
        baseline_text=baseline_text,
        final_text=final_text,
        evidence=ordered,
    )
