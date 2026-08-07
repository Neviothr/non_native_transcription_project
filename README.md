# Transcription Review Workbench

A graphical, semi-automatic system for producing accurate turn-level transcripts of English conversations between a learner and a teacher or AI bot. It follows the supplied project brief and focuses on preserving the learner's actual wording, including grammar errors, hesitations, repetitions, false starts, self-corrections, unclear speech, and switches to Hebrew.

The additional transcription is produced locally with Whisper through `pywhispercpp` and `whisper.cpp`. The application does not use GPT, does not require an API key, and does not upload audio or transcript text to a transcription service.

## Recommended setup

The project is designed for **64-bit Python 3.14.6 on Windows**.

1. Install the official 64-bit Python 3.14.6 release. Keep the default Tcl/Tk component enabled.
2. Extract the project to a normal folder where you have write permission.
3. Double-click `SETUP.bat`.
4. In the graphical setup window, click **Set Up Project**.
5. Click **Launch Application**, or later double-click `RUN.bat`.

The setup creates a private `.venv` folder and installs:

- `pywhispercpp==1.5.0` for local Whisper inference
- `imageio-ffmpeg==0.6.0` for a bundled FFmpeg executable

No command-line knowledge is required. Internet access is required during setup and when a selected Whisper model is downloaded for the first time. After that model is cached, transcription itself can run offline.


## Large recordings and projects

Tab 4 processes evaluation, training-set creation, model training, and exports on a background worker so the Tkinter interface remains responsive. WER and CER are calculated turn by turn with linear-memory edit counting, and Excel transcript rows are streamed directly into the workbook archive. This avoids the project-wide quadratic memory growth that can occur with recordings around 15 minutes or longer.

For an unusually large single turn, the evaluator uses a bounded fallback alignment rather than risking an out-of-memory failure. The evaluation output reports the number of word, character, and source alignments that used this fallback. See `docs/TAB4_SCALING.md` for the implementation details and trade-offs.

## Selective manual review

Version 1.6.20 conditions every Whisper decode window for verbatim disfluency transcription and makes initial final-text selection prefer a near-equivalent source that retains filled pauses, stutters, repetitions, cut-off words, self-corrections, or restarts. The selected source remains unchanged; the pipeline does not synthesize or clean transcript wording.

Version 1.6.19 consolidates adjacent post-transcription turns when their resolved, known speaker is the same. Text, timing, confidence, source evidence, and event associations are preserved in the merged review turn; unresolved `Unknown` turns remain separate.

Version 1.6.18 keeps speech-delay review independent from the general manual-review queue, so detected pauses do not by themselves place otherwise acceptable turns in the filtered review list.

Version 1.6.17 replaces the source-transcript tabs in Review Turns with four stacked, scrollable boxes. The turn table is wider, the right-side transcript editor is smaller, and each source box can be selected as the final-text difference comparison. It also stores non-destructive speech-delay evidence and conservative grammar-sensitive source differences as reviewable structured events. Neither feature changes the literal final transcript automatically.

Version 1.6.16 remembers the last audio and transcript files selected in Project Inputs when the workbench is closed and reopened. It also preserves valid speaker labels from uploaded transcripts verbatim and uses automatic inference only for unlabeled turns.

Version 1.6.13 removes the separate **Add Gold Examples** button and menu action. Eligible examples are collected exclusively and automatically through **Train and Compare ML Models**.

Version 1.6.12 automatically adds eligible, non-duplicate Gold examples when **Train and Compare ML Models** is clicked. Training then uses the combined shared dataset if it contains at least nine examples across at least two quality classes; otherwise the newly eligible examples remain saved and the application reports what is still missing.

Version 1.6.11 highlights word-level differences between every source transcript and the editable final transcript. Source-only or substituted words use a source highlight; final-only or substituted words are highlighted relative to the currently selected source box, updating during editing and source-selection changes.

Version 1.6.10 replaces the indefinite transcription spinner with seven-stage percentage progress as structured status becomes available. The status bar shows the active stage, concise current work, percentage, and elapsed time; high-frequency per-turn updates are throttled while preserving boundaries and completion.

Version 1.6.9 shows the current project filename in the window title and appends `*` while changes are unsaved. Successful manual saves and review autosaves clear the indicator; failed or cancelled saves leave it visible.

Version 1.6.8 adds Previous Review and Next Review navigation. Both actions skip turns outside the review queue, wrap at the ends, save the current editor state before moving, and show the selected item's position within the review queue.

Version 1.6.7 automatically saves review-text, speaker, and speech-flag changes 1.5 seconds after editing stops when the project already has a saved `.ntproject` path. The status bar confirms each autosave with its completion time; manual Save and Save As remain available.

Version 1.6.6 opens a source WAV file once per analysis run and extracts every turn's signal features through that shared handle.

Version 1.6.5 removes the remaining persisted-data compatibility paths. Current project files must match the exact current structure, incompatible training records are rejected instead of migrated, and local Whisper requires callback-delivered segments from the pinned dependency version.

Version 1.6.4 saves its application version in every `.ntproject` file and opens only projects created by that exact version. Older unversioned or differently versioned project files are rejected with a clear compatibility error.

Version 1.6.3 improves analysis performance by avoiding repeated audio-file checks and limiting overlap comparisons to turns whose time ranges can intersect.

Version 1.6.2 displays the application version in the window title and keeps the version and last revision date visible in the status bar.

Version 1.6.1 normalizes source segmentation before quality scoring. A long imported caption can be split across adjacent review turns, and several short captions can be combined into one turn. Each imported word chunk is assigned to at most one turn, so different sentence boundaries do not create duplicated text, artificial source disagreement, or inflated Gold Standard WER labels.

The initial final transcript also counts identical wording from different source slots as separate votes, so two matching sources beat one disagreeing source.

The quality label and manual-review flag are separate. A turn labeled **Needs minor correction** can be left out of the review queue only when its quality score is close to acceptable, at least two transcript sources strongly support the same wording, and no hard-risk condition is present. Empty text, unclear markers, overlapping speech, unresolved speakers, low-consensus differences, and major-correction predictions still require review. Trained ML models also pass through these hard-risk checks.

This policy deliberately favors precision over maximum review reduction. Measure the resulting review rate and Gold Standard WER on your own recordings before changing the thresholds in `src/transcription_app/quality.py`.

## Workflow

### 1. Project Inputs

Enter the learner ID, meeting number, and conversation type. Select the audio file and any existing transcripts. Supported transcript formats are VTT, SRT, TXT, CSV, TSV, and Markdown. ChatGPT and Gold Standard inputs also accept XLSX workbooks.

Zoom VTT files are especially useful because they normally contain timestamps and speaker names. Plain-text transcripts are supported, but alignment is less reliable when neither timestamps nor consistent speaker lines are present.

Imported sources do not need to use identical sentence boundaries. Alignment uses monotonic word evidence together with available timestamps and speaker labels to split or combine imported segments before per-turn agreement and quality features are calculated.

The separate **Import Selected Transcripts** button has been removed. Selecting a transcript with **Browse...** still loads it for immediate inspection, and every click of **Run Local Transcription** reloads all currently selected Zoom, ChatGPT, and Gold Standard files from disk before Whisper starts. This guarantees that edits made to those files outside the application are included in the run. Clearing a transcript path removes that source from the next run.

For XLSX imports, the application searches the first 25 rows of each worksheet for a transcript column. Recognized source-specific headers include `ChatGPT Transcript` and `Gold Standard`; generic headers such as `Transcript`, `Text`, and `Utterance` are also accepted. Optional columns may provide `Start`, `End`, and `Speaker`. A one-column workbook can be imported without a header.

### 2. Local additional transcription

Choose a local multilingual Whisper model and click **Run Local Transcription**.

Available choices:

- `tiny-q5_1`: fastest and least accurate
- `base-q5_1`: light-weight
- `small-q5_1`: recommended default balance
- `medium-q5_0`: more accurate but slower and more memory-intensive
- `large-v3-turbo-q5_0`: strongest offered option, with the largest resource requirements

The selected model downloads once on first use and is stored in the `pywhispercpp` model cache. Audio is converted to 16 kHz mono PCM in a temporary folder, transcribed locally, and then the temporary copy is deleted. Every decode window receives transcript-style context containing filled pauses and false starts so Whisper is less likely to smooth away spoken `um`, `uh`, `er`, `ah`, `eh`, stuttered or cut-off words, repetitions, and sentence restarts. Returned segment text is copied verbatim into the local-model source.

The same prepared WAV is analyzed non-destructively for internal silent pauses before it is deleted. **Detect** enables or disables this analysis, and **Minimum pause (seconds)** controls the candidate threshold (default `0.30`). Absolute start/end times are stored as structured events. A pause inside a turn is shown as `[pause 0.82s]`; silence between different known speakers is shown as `[response gap 0.82s]` before the following turn. Both markers appear in the Review table and delay-aware Excel transcript. Playback for that following row starts at the response gap so it can be checked. **Speech delay reviewed** records an independent review of this evidence. Delay evidence never rewrites the editable literal transcript and does not by itself place an otherwise acceptable turn in the manual-review queue.

The Transcribe tab includes a live **Run time** counter. It starts when **Run Local Transcription** is clicked and stops only after transcript reload, audio preparation, model loading, Whisper inference, source alignment, review-turn creation, and initial quality analysis finish.

The transcription log is deliberately detailed. Every line includes the current clock time and, while a run is active, elapsed run time. It records the selected file and configuration, input and prepared-audio sizes, conversion and model-loading durations, every returned segment with timestamps and a text preview, inference duration and real-time factor, temporary-file cleanup, alignment results, review-turn totals, and complete diagnostic tracebacks if a run fails.

Use `auto` for language detection. The project intentionally offers multilingual model variants rather than English-only variants so that switches into Hebrew have a better chance of remaining visible.

### Speaker handling

Local Whisper supplies timestamped text but does not reliably identify speaker identities. The application therefore uses this priority:

1. If a timed Zoom transcript contains speaker labels, its timestamps and speaker labels define the review turns. The local Whisper text is aligned onto those turns.
2. Any usable speaker label supplied by an uploaded Zoom, Gold Standard, or ChatGPT transcript is preserved verbatim after alignment. For example, `Teacher`, `Dana Cohen`, and `Speaker 2` remain exactly those labels rather than being translated into application-defined roles.
3. Only turns without a usable uploaded label use automatic inference. The fallback considers saved mappings, explicit role words, the configured learner ID, dialogue prompts, speaking activity, conversation type, and names stated in the transcript.
4. When no uploaded transcript labels a turn and the evidence remains ambiguous, its speaker stays `Unknown`. Local Whisper is not represented as speaker diarization.

After speaker labels are finalized, consecutive turns with the same known speaker are consolidated into one review turn. Consecutive `Unknown` turns are not consolidated because a shared placeholder does not establish speaker identity.

Every automatic decision and unresolved label is written to the Transcribe log. This preserves a traceable workflow without presenting Whisper as a speaker-diarization model.

### 3. Review Turns

The review screen shows one row per speaking turn. For every turn, it displays Zoom, ChatGPT, local-model, and Gold Standard text in stacked source boxes. Existing ChatGPT transcripts can still be imported as one comparison source; the project does not generate a new ChatGPT transcript. Detected pause and response-gap markers appear in the turn table, and the editor reports their count, duration, and review status while keeping `final_text` literal and editable. The reviewer can confirm the structured delay evidence independently and can explicitly keep or clear the turn's **Manual review required** state.

You can edit the final transcript, review or correct its speaker label, record special speech features, and mark turns for manual review.

#### Grammar-mistake preservation

The editable `final_text` is the literal transcript layer and is never automatically grammar-corrected or rewritten. Source transcripts remain separate evidence: when the initial transcript or a non-Gold source differs in a narrow grammar-sensitive form, the application records a neutral candidate rather than asserting that either wording is grammatically right or wrong. This separation keeps optional review evidence from silently becoming a corrected transcript.

The precision-first guard covers only conservative differences such as articles, common prepositions, auxiliary or copula forms, pronoun forms, simple inflections, enumerated common irregular verbs, contraction/expansion pairs, and one adjacent word-order swap. A learner turn with an unreviewed candidate remains in the manual-review queue so the reviewer can listen to the audio and choose **Confirmed as spoken**. That action confirms that the literal wording matches the recording; it is not a grammar diagnosis. Candidates are suppressed when overlap, unclear speech, code-switching, self-repair, repetition, a partial word, or another ambiguous context makes the comparison unreliable; an ordinary filler alone does not hide otherwise located evidence.

Grammar-sensitive candidates are saved as structured events with their exact wording, alternate wording, source evidence, text location, pattern, and review decision. They are also included in the Excel **Events** sheet, independently of the literal transcript text.

Use **Split at Final-Text Cursor** when a segment contains two separate turns. Use **Merge with Next** when one turn was divided too aggressively.

### 4. Quality detection and machine learning

Before a trained model exists, the application uses a transparent weighted quality score. It considers transcript agreement, model confidence, source availability, speech rate, WAV signal quality, overlapping speech, unclear markers, and repetition.

The system now stores an immutable `quality_target_text` when a review turn is created. This is the exact unedited transcript candidate whose quality label appears in the Review table. It may come from Zoom, ChatGPT, or the local Whisper model. When sources express the same underlying wording but one retains explicitly detected disfluencies, that verbatim source is preferred even if smoother sources form a majority. Later manual corrections change `final_text` but do not change the ML target.

When **Train and Compare ML Models** is clicked with aligned Gold Standard data, eligible turns are added automatically and labeled from the initial transcript candidate's WER:

- WER up to 0.10: transcript acceptable
- WER above 0.10 and up to 0.30: minor correction
- WER above 0.30: major correction

The training file uses a versioned schema and stable example IDs. Repeated clicks do not duplicate the same turn, while different turns with identical feature values are retained as separate examples. Records that do not match the current schema are rejected.
Saved models also carry target and feature metadata. A classifier from the former target definition is not loaded; add current Gold examples and retrain it.

Click **Train and Compare ML Models** to add eligible Gold examples automatically, then evaluate class-weighted Logistic Regression, class-weighted Linear SVM, class-weighted Random Forest, and a validation-weighted soft-voting ensemble. Model selection uses repeated stratified validation and prioritizes macro F1 and balanced accuracy before ordinary accuracy. This is more reliable than choosing a model from one random holdout, especially when minor- and major-correction examples are less common.

The selected model is saved under `.transcription_support/quality_model.json` and used for later quality flags in that project folder. The model comparison reports accuracy, balanced accuracy, macro F1, selection score, validation prediction count, and the active model.

Small datasets can still produce unstable results. Use Gold Standard turns from multiple sessions and include examples from all three quality classes before treating the classifier as reliable. See `docs/ML_QUALITY_LABELS.md` for the target definition and validation method.

### 5. Evaluation and export

The application calculates:

- Word Error Rate and Character Error Rate
- substitutions, deletions, and insertions
- speaker-identification accuracy when usable Gold Standard speaker labels are available; otherwise the metric is reported as N/A
- location-aware precision, recall/preservation, and F1 for transparent fillers, partial words, repeated phrases, self-corrections, unclear markers, and Hebrew words; otherwise the metric is reported as N/A
- annotated grammar-error token preservation and substitution/deletion loss rates when Gold Standard tokens explicitly use the `@!` suffix; otherwise these metrics are reported as N/A
- manual-review rate

Grammar preservation is evaluated only from explicit Gold annotations. For example, `have@!` marks the exact Gold token `have` as wording that must remain literal. The evaluator does not infer grammar mistakes from unannotated Gold text. Any substitution or deletion of an annotated token counts as loss; the metric does not claim that every loss was an intentional grammar correction.

**Export Excel** creates an `.xlsx` workbook with six sheets:

- Transcript
- Evaluation
- Source Comparison
- ML Model Comparison
- Metadata
- Events

The Transcript sheet contains both literal **Final Transcript** and rendered **Final Transcript with Delays** columns. The Events sheet retains speech-delay and grammar-sensitive evidence, including event type, absolute timing where applicable, source, review state, token position, and detector details.

**Export HTML Report** creates a self-contained report with tables and SVG charts.

## Signal features

Silent-pause detection uses the prepared 16 kHz PCM WAV and therefore works for every supported input format. Volume and estimated signal-to-noise ratio are still calculated directly only for uncompressed PCM WAV source files; those two fields remain blank for other original formats.

## Project files

- `main.py` launches the application.
- `setup_gui.py` creates the virtual environment and installs packages through a GUI.
- `requirements.txt` pins all direct third-party requirements.
- `src/transcription_app/gui.py` contains the Tkinter interface.
- `src/transcription_app/local_whisper.py` performs local audio conversion and Whisper transcription.
- `src/transcription_app/parsers.py` imports existing transcripts.
- `src/transcription_app/alignment.py` aligns transcript sources.
- `src/transcription_app/quality.py` extracts review-quality features.
- `src/transcription_app/ml_models.py` implements the three classifiers.
- `src/transcription_app/evaluation.py` calculates Gold Standard metrics.
- `src/transcription_app/xlsx_writer.py` creates the Excel workbook without an Excel library.
- `src/transcription_app/reporting.py` creates the HTML and SVG evaluation report.
- `docs/REQUIREMENTS_MAPPING.md` maps project requirements to the implementation.

## Tests

After setup, double-click `RUN_TESTS.bat`. Tests create temporary files only and do not include a demonstration dataset.

## Known limitations and trade-offs

Whisper can still normalize or omit some disfluencies, especially quiet filler sounds, repetitions, and unclear fragments. The speech-error preservation metric measures only transparently detectable transcript events. The grammar-preservation guard is similarly precision-first: it surfaces a small set of source disagreements, not a complete grammar diagnosis, and it never rewrites the transcript. The final review stage remains necessary because the research target is stricter than ordinary readable transcription.

The grammar guard intentionally does not classify every non-native construction. It considers only narrow enumerated surface patterns—including selected function words, inflections, irregular verbs, contractions, and a single adjacent swap—and suppresses candidates in ambiguous, overlapping, unclear, and code-switched turns. Its evaluation metrics require explicit `@!` Gold annotations and remain N/A when no annotated grammar-error tokens are present.

Silent-pause detection is frame-energy based rather than a word-level forced aligner. Automatic marker position within a turn is estimated from relative time, is identified as estimated in event details, and must be checked against audio. The detector distinguishes acoustic silence from transcript text but cannot by itself decide whether a quiet interval expresses hesitation, deliberate emphasis, or recording conditions.

The local model does not solve speaker identification by itself. Timed Zoom speaker labels remain the preferred scaffold. The automatic mapper uses transcript labels and alignment evidence, not voice biometrics or neural diarization. Labels without defensible evidence remain `Unknown` and require correction in Review Turns. Adding a modern neural diarization stack would substantially increase package size, setup complexity, and hardware requirements, and should be evaluated as a separate project extension rather than silently presented as reliable in this baseline.

## Turn-level audio review

Each row in the **Review Turns** table contains an **Audio** cell. Click **▶ Play** to extract and play only that turn's start-to-end range from the selected original recording. Click **■ Stop**, click the same row again, or use the toolbar's **Stop Playback** button to stop it. Playback uses the existing bundled FFmpeg package plus the Windows standard-library audio player, so no additional package is required.
