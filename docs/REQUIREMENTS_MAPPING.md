# Requirements Mapping

| Project requirement | Implementation |
|---|---|
| Load audio and existing transcripts | Project Inputs tab accepts audio plus Zoom, ChatGPT, and Gold Standard files. Parsers support VTT, SRT, TXT, CSV, TSV, and Markdown; ChatGPT and Gold Standard inputs also support XLSX workbooks without adding a spreadsheet dependency. All selected transcript files are reloaded from disk whenever local transcription starts. |
| Run an additional transcription model | Transcribe tab runs a local multilingual Whisper model through `pywhispercpp` and `whisper.cpp`. No GPT model or transcription API is called. |
| Easy package setup | `SETUP.bat` opens a Tkinter setup window that creates `.venv` and installs pinned packages from `requirements.txt`. |
| Support common audio formats | `imageio-ffmpeg` supplies a bundled FFmpeg executable and converts input audio to temporary 16 kHz mono PCM WAV before local inference. |
| Match text to the timeline | Local Whisper segments include timestamps. Alignment prefers time overlap and falls back to monotonic text similarity. |
| Divide the conversation into speaking turns | A timed labeled transcript is used as the turn scaffold when available. Otherwise local Whisper timestamped segments become turns and speaker labels from an untimed Zoom, Gold, or ChatGPT transcript are transferred by text alignment. |
| Identify conversation-specific speaker identities | WebVTT voice tags are preserved. AI conversations use AI and optional Supervisor roles; human-teacher conversations use Teacher. The learner remains Student only when no defensible name is found. Actual human names are retained from transcript labels or extracted independently from final, Zoom, ChatGPT, local-model, and Gold Standard text. A self-declared name becomes the project-wide learner identity and is propagated to all matching student turns, even when later turns reuse an Unknown raw label. Mapping also uses saved identities, learner-ID matches, dialogue prompts, speaking activity, and remaining-role elimination. Every decision is written to the transcription log. |
| Compare Zoom, ChatGPT, and another model | Each turn stores Zoom, imported ChatGPT, and local Whisper text and calculates agreement and pairwise similarity features. |
| Identify unreliable segments | Transparent quality scoring and optional trained classifiers use agreement, confidence, source differences, speech features, overlap, and disfluency indicators. |
| Mark segments for manual inspection | Every turn receives a quality label and a separate manual-review decision. Near-boundary minor turns are auto-cleared only when multiple sources strongly agree and no hard-risk condition exists; unclear, overlapping, empty, low-consensus, major-correction, and unresolved-speaker turns remain queued. |
| Review the original recording per turn | Every row in Review Turns includes an Audio cell that extracts and plays only that turn's timestamp range from the selected recording. |
| Produce one Excel file | The built-in XLSX writer exports transcript, evaluation, source-comparison, model-comparison, and metadata sheets. |
| Preserve grammar errors and disfluencies | Review fields detect and retain repetitions, hesitation markers, self-correction patterns, unclear markers, and Hebrew characters without grammar correction. The reviewer remains the final authority. |
| Evaluate against Gold Standard | Evaluation includes WER, CER, substitutions, deletions, insertions, speaker accuracy, and event-level preservation measures. Metrics without a valid Gold Standard denominator are reported as N/A rather than zero. Manual correction-time tracking is intentionally not included. |
| Compare ML models | Pure-Python Logistic Regression, Linear SVM, and Random Forest implementations are trained and compared using held-out macro F1. |
| Produce graphs and tables | HTML report contains metric tables and self-contained SVG charts. Excel includes evaluation tables. |
| No demo or demo data | The project contains no sample recordings, generated transcript rows, or synthetic training dataset. Tests use temporary data only. |

## Important boundary

The local Whisper model is a transcription model, not a dependable speaker-identification system. The application therefore obtains speaker identities from imported transcript labels, including Zoom WebVTT voice tags, and transfers untimed labels by text alignment. For ordinary two- or three-person conversations, unresolved role names are completed with logged dialogue-role heuristics. If no imported transcript contains speaker labels, true speaker separation still requires a diarization-capable model.
