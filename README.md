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

The selected model downloads once on first use and is stored in the `pywhispercpp` model cache. Audio is converted to 16 kHz mono PCM in a temporary folder, transcribed locally, and then the temporary copy is deleted.

The Transcribe tab includes a live **Run time** counter. It starts when **Run Local Transcription** is clicked and stops only after transcript reload, audio preparation, model loading, Whisper inference, source alignment, review-turn creation, and initial quality analysis finish.

The transcription log is deliberately detailed. Every line includes the current clock time and, while a run is active, elapsed run time. It records the selected file and configuration, input and prepared-audio sizes, conversion and model-loading durations, every returned segment with timestamps and a text preview, inference duration and real-time factor, temporary-file cleanup, alignment results, review-turn totals, and complete diagnostic tracebacks if a run fails.

Use `auto` for language detection. The project intentionally offers multilingual model variants rather than English-only variants so that switches into Hebrew have a better chance of remaining visible.

### Speaker handling

Local Whisper supplies timestamped text but does not reliably identify speaker identities. The application therefore uses this priority:

1. If a timed Zoom transcript contains speaker labels, its timestamps and speaker labels define the review turns. The local Whisper text is aligned onto those turns.
2. The selected conversation type constrains the fixed roles. AI conversations use `AI` and may include `Supervisor`; human-teacher conversations use `Teacher`. The learner is temporarily represented as `Student` only when no reliable name is available.
3. The application automatically maps raw labels using saved mappings, explicit role words, the configured learner ID, aligned Gold Standard or ChatGPT speaker evidence, dialogue prompts, and speaking activity. It searches the final, Zoom, ChatGPT, local-model, and Gold Standard text independently for explicit learner introductions. A supported name becomes the project-wide learner identity and is propagated to every matching student turn, including later turns that reuse the same `Unknown` raw-speaker placeholder. Legacy `Learner` values become `Student`; a `Teacher` label in an AI conversation becomes `Supervisor`.
4. When all but one allowed participant role is known, the only remaining role may be assigned by elimination. Ambiguous labels remain `Unknown` rather than being forced into a disallowed role. Detected names are propagated to aligned sources so Gold Standard speaker evaluation uses the same identity.

Every automatic decision and unresolved label is written to the Transcribe log. This preserves a traceable workflow without presenting Whisper as a speaker-diarization model.

### 3. Review Turns

The review screen shows one row per speaking turn. For every turn, it displays Zoom, ChatGPT, local-model, and Gold Standard text side by side in tabs. Existing ChatGPT transcripts can still be imported as one comparison source; the project does not generate a new ChatGPT transcript.

You can edit the final transcript, correct the speaker identity from the conversation-type-specific roles and detected learner names, record special speech features, and mark turns for manual review.

Use **Split at Final-Text Cursor** when a segment contains two separate turns. Use **Merge with Next** when one turn was divided too aggressively.

### 4. Quality detection and machine learning

Before a trained model exists, the application uses a transparent weighted quality score. It considers transcript agreement, model confidence, source availability, speech rate, WAV signal quality, overlapping speech, unclear markers, and repetition.

The system now stores an immutable `quality_target_text` when a review turn is created. This is the exact unedited transcript candidate whose quality label appears in the Review table. It may come from Zoom, ChatGPT, or the local Whisper model. Later manual corrections change `final_text` but do not change the ML target.

With aligned Gold Standard data, click **Add Gold Examples**. Each turn is labeled from the initial transcript candidate's WER:

- WER up to 0.10: transcript acceptable
- WER above 0.10 and up to 0.30: minor correction
- WER above 0.30: major correction

The training file uses a versioned schema and stable example IDs. Repeated clicks do not duplicate the same turn, while different turns with identical feature values are retained as separate examples. Legacy records labeled only from local-Whisper WER are discarded when the training set is rebuilt because they answer a different prediction question.
Saved models also carry target and feature metadata. A classifier from the former target definition is not loaded; add current Gold examples and retrain it.

Click **Train and Compare ML Models** to evaluate class-weighted Logistic Regression, class-weighted Linear SVM, class-weighted Random Forest, and a validation-weighted soft-voting ensemble. Model selection uses repeated stratified validation and prioritizes macro F1 and balanced accuracy before ordinary accuracy. This is more reliable than choosing a model from one random holdout, especially when minor- and major-correction examples are less common.

The selected model is saved under `.transcription_support/quality_model.json` and used for later quality flags in that project folder. The model comparison reports accuracy, balanced accuracy, macro F1, selection score, validation prediction count, and the active model.

Small datasets can still produce unstable results. Use Gold Standard turns from multiple sessions and include examples from all three quality classes before treating the classifier as reliable. See `docs/ML_QUALITY_LABELS.md` for the target definition, validation method, and migration behavior.

### 5. Evaluation and export

The application calculates:

- Word Error Rate and Character Error Rate
- substitutions, deletions, and insertions
- speaker-identification accuracy when usable Gold Standard speaker labels are available; otherwise the metric is reported as N/A
- event-level preservation rate for detected hesitations, repetitions, self-corrections, unclear markers, and Hebrew words; otherwise the metric is reported as N/A
- manual-review rate

**Export Excel** creates an `.xlsx` workbook with five sheets:

- Transcript
- Evaluation
- Source Comparison
- ML Model Comparison
- Metadata

**Export HTML Report** creates a self-contained report with tables and SVG charts.

## Signal features

Volume and estimated signal-to-noise ratio are calculated directly only for uncompressed PCM WAV source files. Other audio formats can still be converted, transcribed, aligned, reviewed, evaluated, and exported, but those source-signal fields remain blank unless the original input is a compatible WAV file.

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

Whisper can still normalize or omit some disfluencies, especially quiet filler sounds, repetitions, and unclear fragments. The speech-error preservation metric measures only transparently detectable transcript events; it does not infer grammatical errors. The final review stage remains necessary because the research target is stricter than ordinary readable transcription.

The local model does not solve speaker identification by itself. Timed Zoom speaker labels remain the preferred scaffold. The automatic mapper uses transcript labels and alignment evidence, not voice biometrics or neural diarization. Labels without defensible evidence remain `Unknown` and require correction in Review Turns. Adding a modern neural diarization stack would substantially increase package size, setup complexity, and hardware requirements, and should be evaluated as a separate project extension rather than silently presented as reliable in this baseline.

## Turn-level audio review

Each row in the **Review Turns** table contains an **Audio** cell. Click **▶ Play** to extract and play only that turn's start-to-end range from the selected original recording. Click **■ Stop**, click the same row again, or use the toolbar's **Stop Playback** button to stop it. Playback uses the existing bundled FFmpeg package plus the Windows standard-library audio player, so no additional package is required.
