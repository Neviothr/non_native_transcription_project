# Speaker and Speech-Error Metric Fix

Version 1.5.1 corrects misleading zero values in Gold Standard evaluation.

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
