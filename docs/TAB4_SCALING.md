# Tab 4 Scaling and Stability

## Root cause

The previous evaluation implementation allocated two complete dynamic-programming matrices for every WER or CER calculation. Tab 4 also joined every Gold Standard turn into one project-wide string before character evaluation. Memory therefore grew quadratically with transcript length, and a long recording could terminate Python rather than returning a normal exception.

Excel export also assembled the entire Transcript worksheet as one XML string before writing the workbook.

## Changes in version 1.5.12

- WER and CER use linear-memory edit counting.
- Project metrics are aggregated per aligned turn instead of evaluating one combined transcript matrix.
- Oversized individual turns use a bounded `SequenceMatcher` fallback. The output records how many word, character, and source alignments required this approximation.
- Calculate Evaluation, Add Gold Examples, Train and Compare ML Models, Export Excel, and Export HTML Report run outside Tk's main thread.
- Tab 4 prevents overlapping operations and shows its own progress indicator.
- Excel rows are streamed directly into the ZIP archive and the workbook is replaced atomically only after a successful write.
- Training-set JSON is also replaced atomically.
- Model comparison uses at most 5,000 class-balanced records, adaptive epoch budgets, and at most 2,500 bootstrap rows per random-forest tree.

## Trade-offs

Turn-level aggregation assumes that the imported Gold Standard and source transcripts are already aligned to the same turn boundaries. This is the correct assumption for the application's review workflow and prevents edits in one turn from being matched against words in a neighboring turn.

The bounded fallback is not guaranteed to produce the same edit path as full Levenshtein dynamic programming. It is used only when a single alignment would exceed 2,000,000 matrix cells. The approximation counters make this visible instead of silently presenting the result as exact.

Excel limits one cell to 32,767 characters. A pathological single-turn transcript longer than this is exported with an explicit truncation marker; the project file remains unchanged.
