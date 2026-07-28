# Requirements Mapping

| Project requirement | Implementation |
|---|---|
| Load audio and existing transcripts | Project Inputs tab accepts audio plus Zoom, ChatGPT, and Gold Standard files. Parsers support VTT, SRT, TXT, CSV, TSV, and Markdown. |
| Run an additional transcription model | Transcribe tab runs a local multilingual Whisper model through `pywhispercpp` and `whisper.cpp`. No GPT model or transcription API is called. |
| Easy package setup | `SETUP.bat` opens a Tkinter setup window that creates `.venv` and installs pinned packages from `requirements.txt`. |
| Support common audio formats | `imageio-ffmpeg` supplies a bundled FFmpeg executable and converts input audio to temporary 16 kHz mono PCM WAV before local inference. |
| Match text to the timeline | Local Whisper segments include timestamps. Alignment prefers time overlap and falls back to monotonic text similarity. |
| Divide the conversation into speaking turns | A timed Zoom transcript with speaker labels is used as the turn scaffold when available. Otherwise local Whisper timestamped segments become provisional turns. |
| Identify Learner, Teacher, and Supervisor | Existing transcript speaker labels are inferred and mapped in the GUI. Unknown labels remain explicit for manual mapping rather than being guessed. |
| Compare Zoom, ChatGPT, and another model | Each turn stores Zoom, imported ChatGPT, and local Whisper text and calculates agreement and pairwise similarity features. |
| Identify unreliable segments | Transparent quality scoring and optional trained classifiers use agreement, confidence, source differences, speech features, overlap, and disfluency indicators. |
| Mark segments for manual inspection | Every turn receives a quality label and manual-review flag, with a filtered review view. |
| Review the original recording per turn | Every row in Review Turns includes an Audio cell that extracts and plays only that turn's timestamp range from the selected recording. |
| Produce one Excel file | The built-in XLSX writer exports transcript, evaluation, source-comparison, model-comparison, and metadata sheets. |
| Preserve grammar errors and disfluencies | Review fields detect and retain repetitions, hesitation markers, self-correction patterns, unclear markers, and Hebrew characters without grammar correction. The reviewer remains the final authority. |
| Evaluate against Gold Standard | Evaluation includes WER, CER, substitutions, deletions, insertions, speaker accuracy, and preservation measures. Manual correction-time tracking is intentionally not included. |
| Compare ML models | Pure-Python Logistic Regression, Linear SVM, and Random Forest implementations are trained and compared using held-out macro F1. |
| Produce graphs and tables | HTML report contains metric tables and self-contained SVG charts. Excel includes evaluation tables. |
| No demo or demo data | The project contains no sample recordings, generated transcript rows, or synthetic training dataset. Tests use temporary data only. |

## Important boundary

The local Whisper model is a transcription model, not a dependable speaker-identification system. The baseline uses timed Zoom speaker labels when available and manual mapping otherwise. This limitation is exposed in the interface and documentation instead of presenting `Unknown` speaker assignments as automatic diarization.
