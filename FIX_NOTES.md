# Version 1.6.8

- Adds Previous Review navigation alongside Next Review.
- Skips unflagged turns and wraps at both ends of the review queue.
- Saves current editor values before navigating.
- Displays the selected turn's position and total review-item count.

# Version 1.6.7

- Autosaves review transcript, speaker, and speech-flag edits after a 1.5-second idle period.
- Autosaves only after the project has an established `.ntproject` path, avoiding unexpected Save As dialogs.
- Shows autosave completion time or a recoverable failure message in the status bar.
- Cancels pending autosave callbacks during manual save, project replacement, and application shutdown.

# Version 1.6.6

- Opens a WAV source once while extracting volume and noise features for all turns.
- Retains the single-interval audio-analysis API through the new batch implementation.
- Adds regression coverage verifying that multiple intervals use one file open.

# Version 1.6.5

- Requires exact current fields throughout `.ntproject` data instead of ignoring unknown or missing fields.
- Removes legacy quality-target reconstruction and the historical `Learner` role alias.
- Rejects incompatible quality-training records instead of silently replacing them.
- Removes the alternate Whisper return-value path and requires callback-delivered segment copies.

# Version 1.6.4

- Records the current application version in newly saved `.ntproject` files.
- Rejects unversioned projects and projects saved by any other application version.
- Removes implicit loading of legacy `.ntproject` structures.

# Version 1.6.3

- Reduces overlap analysis work by comparing only turns with potentially intersecting time ranges.
- Checks audio-file eligibility once per analysis run instead of once per turn.
- Updates the application revision date whenever the application version changes.

# Version 1.6.2

- Displays the application version in the main window title.
- Adds a persistent status-bar label with the version and last revision date.
- Stores the revision date beside the package version so the GUI uses one shared metadata source.
- Adds regression coverage for the displayed release metadata.

# Version 1.6.1

- Normalizes transcript boundaries before quality scoring and Gold Standard labeling.
- Splits a long imported segment across adjacent review turns using monotonic word, timestamp, and speaker evidence.
- Combines multiple short imported segments when they belong to one review turn.
- Assigns every imported word chunk to at most one turn, preventing duplicated text from lowering source agreement.
- Allows the same original segment to provide speaker or confidence metadata to multiple turns after its text is split.
- Advances the quality-training schema to version 4 so models trained from pre-normalization labels are rejected and retrained.
- Adds regression tests for timed splitting, segment combining, untimed boundary differences, quality features, final quality labels, and Gold training labels.

# Version 1.6.0

- Adds an immutable `quality_target_text` so each ML label refers to the unedited transcript initially shown in Review Turns.
- Labels Gold examples from the selected initial transcript's WER instead of always using local-Whisper WER.
- Prevents manually corrected `final_text` from leaking into lexical quality features or training targets.
- Migrates the quality-training JSON to a versioned schema and removes incompatible legacy labels when examples are rebuilt.
- Uses stable example IDs so repeated clicks do not duplicate one turn, while distinct turns with identical feature vectors remain available for training.
- Adds inverse-frequency class weighting to Logistic Regression, Linear SVM, and Random Forest.
- Replaces single-split model selection with repeated stratified validation using macro F1, balanced accuracy, and accuracy.
- Evaluates and serializes a validation-weighted soft-voting ensemble alongside the three individual classifiers.
- Expands GUI and Excel model comparisons with balanced accuracy, selection score, validation count, and active-model status.
- Adds regression tests for target-source correctness, manual-edit leakage, legacy reconstruction, schema migration, and ensemble persistence.

# Version 1.5.13

- Fixes initial transcript selection so matching Zoom and ChatGPT wording counts as two independent votes instead of being discarded as a duplicate string.
- Separates the three-class quality label from the binary manual-review decision.
- Automatically clears near-boundary minor-correction turns only when at least two transcript sources provide strong consensus and the turn has no hard-risk signal.
- Keeps empty text, unclear markers, overlapping speech, unresolved speakers, low-consensus turns, and major-correction predictions in the manual-review queue.
- Prevents an optimistic ML prediction from bypassing hard-risk checks.
- Validates and normalizes classifier probability output before using it.
- Adds six regression tests covering majority selection, consensus auto-clear, disagreement retention, hard-risk overrides, ML boundary handling, and unresolved speakers.

# Version 1.5.12

- Replaces the project-wide quadratic WER/CER matrices with turn-level, linear-memory edit counting.
- Adds a bounded SequenceMatcher fallback for unusually large individual turns and reports when an approximation was required.
- Runs Calculate Evaluation, Add Gold Examples, Excel export, HTML export, and model training through a dedicated Tab 4 worker boundary.
- Disables Tab 4 action buttons while one operation is active and shows a dedicated progress bar, preventing overlapping writes and Tk re-entrancy.
- Creates a detached project snapshot before background evaluation/export so workers never read or update Tk widgets.
- Streams transcript worksheet rows directly into the XLSX ZIP archive instead of building the complete worksheet XML in RAM.
- Enforces Excel's 32,767-character cell limit with an explicit truncation marker for pathological single-turn cells.
- Adds large-project regression tests covering 600-turn evaluation memory and a 2,000-turn streamed Excel export.

# Version 1.5.11

- Opens saved projects on a dedicated worker thread and sends every progress update back through the existing Tk main-thread dispatch queue.
- Shows determinate percentage progress and detailed timestamped project-open logging in the Transcribe tab.
- Restores saved turns, source segments, mappings, metrics, and alignments directly instead of reloading and quadratically re-aligning external transcripts during every open.
- Leaves the current project unchanged when opening fails and writes the complete traceback to the process log.
- Reports the project size, turn count, saved source counts, quality-model status, and missing referenced input files.
- Adds clear project-file validation errors for missing files, invalid UTF-8, malformed JSON, and incompatible structures.
- Ignores unknown fields from older or newer compatible project versions rather than rejecting the whole project.
- Caps the shared Review Treeview row height so one abnormally long turn cannot exhaust native Tk resources while the project is rendered.
- Adds regression coverage for asynchronous opening, progress/logging, direct saved-state restoration, compatibility handling, malformed JSON, and extreme row heights.

# Version 1.5.10

- Routes every transcription-worker status update and completion callback through a thread-safe queue that is drained only by Tk's main thread.
- Removes worker-thread calls to `Tk.after`, which can corrupt Tcl/Tk state after native Whisper callbacks and cause a later Tab 4 button press to terminate the application.
- Removes forced `update_idletasks()` calls from normal status and evaluation logging to avoid unnecessary nested Tk event processing.
- Adds regression coverage for main-thread dispatch and the background-operation boundary.

# Tab 4 Process Timer

- Adds a live `Process time` display beside the Evaluate and Export process-log heading.
- Starts the timer whenever Calculate Evaluation, Add Gold Examples, Train and Compare ML Models, Export HTML Report, or Export Excel is pressed.
- Updates the visible timer every 0.1 seconds while an operation is active and preserves the final duration with its completed, cancelled, stopped, or failed outcome.
- Adds elapsed `+HH:MM:SS.t` prefixes to Tab 4 process-log entries.
- Cancels the scheduled timer callback safely when the application closes.
- Adds regression coverage for the timer widget, elapsed log prefixes, and start/stop behavior.

# Version 1.5.9

- Replaces the free-text language-code field with a read-only dropdown.
- Keeps `auto` as the default selection.
- Lists every configured Whisper language as `code (language name)`.
- Includes `yue (Cantonese)` with a note that it requires `large-v3-turbo-q5_0`.
- Converts the displayed dropdown label back to the raw language code before starting transcription.
- Adds regression tests for the complete language list, default selection, widget type, and code extraction.

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
