# Machine-Learning Quality Labels

## Prediction target

The Quality column predicts the correction level of the transcript candidate initially shown to the reviewer. That candidate is stored in `Turn.quality_target_text` before manual editing.

This distinction matters because the selected candidate may come from Zoom, ChatGPT, or local Whisper. Training every example from local-Whisper WER teaches a different target whenever another source supplied the displayed transcript. Training from the editable final transcript is also invalid because manual corrections leak the answer into the training data.

For a newly created turn without `quality_target_text`, the application selects the strongest current source candidate and stores it before review so later manual edits cannot change it.

## Segment-boundary normalization

Imported systems do not have to use the same sentence or caption boundaries. Before quality features or Gold Standard WER labels are calculated, the alignment layer assigns ordered source word chunks to review turns. This permits:

- one long imported segment to be split across adjacent turns;
- several short imported segments to be combined into one turn;
- timed and untimed transcripts to use the same monotonic text flow;
- every imported word chunk to contribute to at most one turn, preventing duplicated source text.

As a result, the quality model measures wording disagreement rather than differences in segmentation.

## Gold Standard labels

Each eligible turn compares `quality_target_text` with `gold_text`.

| Initial-transcript WER | Label |
|---|---|
| `<= 0.10` | Transcript acceptable |
| `> 0.10` and `<= 0.30` | Needs minor correction |
| `> 0.30` | Needs major correction |

The thresholds remain explicit and auditable. They can be changed later after measuring reviewer effort and label agreement on the actual study data.

## Training-set schema

`quality_training.json` uses a versioned record schema. Each record stores:

- the feature vector and feature names;
- the three-class label;
- the initial-transcript WER;
- the label-target identifier;
- a stable SHA-256 example ID.

The example ID is based on project metadata, turn timing, source transcripts, the preserved quality target, and Gold text. This prevents repeated button clicks from adding the same turn again. It does not deduplicate solely by feature vector, because two legitimate turns may have identical numerical features and both should contribute to training.

**Train and Compare ML Models** automatically appends eligible examples from the current project before checking the combined dataset and starting training.

Records that do not match the current target definition are rejected. Mixing different label definitions would introduce contradictory supervision.

Saved model files include the same target, schema, and feature metadata. An older model without matching metadata is rejected instead of silently producing labels for the wrong target.

## Model comparison

The implementation remains dependency-free and compatible with the project's Python setup. It evaluates:

- class-weighted multinomial Logistic Regression;
- class-weighted one-vs-rest Linear SVM;
- class-weighted Random Forest;
- a soft-voting ensemble weighted by validation performance.

Training data is capped with deterministic class-balanced sampling for large datasets. Model comparison uses repeated stratified holdouts rather than one split. Selection prioritizes:

- macro F1, to avoid hiding weak minority-class performance;
- balanced accuracy, to weight each represented class equally;
- ordinary accuracy, as a secondary measure.

The weighted ensemble is selected only when its validation score is at least as strong as the individual candidates.

## Manual-review safeguards

ML predicts the quality label, but hard review rules remain independent. Empty text, unclear speech, overlapping speech, and unresolved speakers still require manual review even when the model predicts an acceptable transcript.

This separation is intentional. A classifier can estimate correction severity, but it should not override conditions that make automatic clearance unsafe or unauditable.
