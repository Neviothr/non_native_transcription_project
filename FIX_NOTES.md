# Version 1.5.8

- Treats a reliable learner-name extraction as a project-wide identity instead of a one-turn display value.
- Propagates the detected name to every turn already marked `Student` and to later turns sharing the same raw speaker label, including repeated `Unknown` placeholders.
- Keeps `AI`, `Teacher`, and `Supervisor` turns unchanged during propagation.
- Runs learner-name propagation before the Review Turns table is rendered, so all matching rows update together without requiring row-by-row selection.
- Logs the detected learner name and the number of turns updated.
- Adds regression tests for repeated unknown learner turns in both AI and human-teacher conversations.

# Version 1.5.7

- Fixed the Review Turns speaker field showing `Unknown` when a learner name was extracted from a turn whose raw speaker label was `Unknown`.
- Name extraction now checks final, Zoom, ChatGPT, local-model, and Gold Standard text independently instead of only the first available transcript version.
- Speaker mapping lookups now tolerate harmless whitespace and capitalization differences in raw labels.
- Added regression tests for unknown raw labels, multi-source name extraction, and whitespace-normalized mappings.

# Version 1.5.6

- Preserves an actual learner name instead of replacing it with `Student` when a non-generic transcript speaker label contains a human name.
- Extracts learner names from explicit transcript introductions such as `My name is Maya` and from repeated direct-address evidence.
- Keeps `AI`, `Teacher`, and `Supervisor` as fixed role labels; only the non-facilitator human participant is replaced with a detected name.
- Propagates the final identity back to aligned Zoom, ChatGPT, and Gold Standard labels so speaker evaluation remains consistent.
- Updates the Review Turns speaker selector so detected names remain visible and are not normalized back to `Unknown`.
- Adds regression coverage for AI and human-teacher conversations, named labels, self-introductions, and aligned named Gold Standard labels.

# Version 1.5.5

- Makes speaker roles depend on the selected conversation type.
- AI conversations use only `Student`, `Supervisor`, and `AI`; the supervisor is treated as the human teacher-like participant and is optional.
- Human-teacher conversations use only `Student` and `Teacher`.
- Updates the Review Turns role selector immediately when the conversation type changes and automatically remaps existing turns.
- Migrates legacy `Learner` mappings to `Student`, maps `Teacher` to `Supervisor` in AI conversations, and maps legacy `Supervisor` to `Teacher` in human-teacher conversations.
- Logs the conversation-type role constraints and every resulting automatic mapping decision.

## Version 1.5.4

- Preserves Zoom WebVTT `<v Speaker Name>` voice labels instead of stripping them as markup.
- Uses the best labeled Zoom, Gold Standard, or ChatGPT source as the speaker scaffold.
- Transfers untimed transcript speaker labels onto timestamped Whisper turns through monotonic alignment.
- Completes ordinary two- and three-speaker role mappings with logged dialogue-role fallbacks, so participant names and `Speaker 1/2` no longer remain `Unknown` by default.
- Keeps `Unknown` only when no imported transcript provides any speaker labels, because local Whisper is not a diarization model.
- Recovers all-`Unknown` rows in existing saved projects when their original transcript files are still available.

## Version 1.5.3: Automatic Speaker Mapping

- Removed the **Map Speakers** button and Tools-menu command.
- Speaker mapping now runs automatically during Stage 6 after transcript alignment.
- Mapping uses saved project mappings, explicit role labels, learner-ID matches, aligned Gold/ChatGPT role evidence, and limited role elimination.
- Every resolved and unresolved label is written to the Transcribe log.
- Ambiguous labels remain `Unknown` and can still be corrected in Review Turns.

# Release Notes

## Version 1.5.2: Transcript Reload on Run

- Removed the **Import Selected Transcripts** button and its GUI handler.
- **Run Local Transcription** now reloads the selected Zoom, ChatGPT, and Gold Standard files from disk before local Whisper inference starts.
- Empty transcript path fields remove previously loaded stale sources from the run.
- Transcript reload is atomic. If any selected transcript cannot be parsed, the run stops before Whisper starts and the previously imported source data is not partially replaced.
- The transcription log and run timer now include the reload stage.
- Added regression tests for button removal, reload ordering, stale-source removal, and failure atomicity.

## Version 1.5.1: Speaker and Speech-Error Metric Fix

## Changes

- Speaker Accuracy is now `N/A` when the Gold Standard has no usable speaker labels.
- `Unknown`, blank, and similar placeholder labels are not treated as valid Gold Standard speakers.
- A known Gold Standard speaker with an `Unknown` predicted speaker still counts as incorrect.
- Common role aliases such as Student/Learner and Tutor/Teacher are compared canonically.
- Speech Error Preservation Rate is now event-based rather than turn-Boolean-based.
- Detectable events include hesitation markers, adjacent repetitions, explicit self-corrections, unclear markers, and Hebrew words.
- The preservation metric is `N/A` when the Gold Standard contains no detectable events.
- Evaluation output includes denominator and numerator counts for both metrics.
- GUI, Excel, and HTML exports display unavailable metrics as `N/A`.

The speech-error detector does not infer grammatical errors. Grammar-error preservation still requires explicit annotation or a separate grammar-aware evaluation method.
