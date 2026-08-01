# ML Transcript Enhancement

## What changed

The application now trains a second machine-learning task in addition to transcript-quality classification. The new task learns which aligned transcript source is most likely to be closest to the spoken turn.

Candidates are limited to:

- raw local Whisper text
- imported ChatGPT text
- imported Zoom text

The aligned Gold Standard is used only to label the best source during training. It is never used as an inference candidate.

## Runtime behavior

For each turn, the enhancer extracts source availability, Whisper confidence, pairwise similarity, consensus, relative length, repetition, unclear-marker, Hebrew-switch, and self-correction features. The selected Logistic Regression, Linear SVM, or Random Forest model predicts the strongest available source.

The selected text is stored verbatim in `Turn.enhanced_text`, with provenance in:

- `enhancement_source`
- `enhancement_confidence`
- `enhancement_method`

The raw `model_text` remains unchanged. The final transcript is replaced only when it is empty or still matches an automatic source candidate. Text that differs from all source candidates is treated as a reviewer edit and preserved.

## Training files

The project support directory can now contain:

- `quality_training.json`
- `quality_model.json`
- `transcript_enhancement_training.json`
- `transcript_enhancer.json`

The enhancement model requires at least 9 multi-source examples and at least 2 winning source classes.

## Verification

The repository contains 100 passing unit tests, including six new transcript-enhancement tests. A synthetic end-to-end check also verified model training, application, project save/load, Excel export, HTML export, and Gold-based source comparison.
