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
