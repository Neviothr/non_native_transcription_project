"""Local Whisper transcription through pywhispercpp.

The speech model runs on the user's computer. No audio or transcript text is sent
through an API. Audio is normalized to 16 kHz mono PCM with the FFmpeg binary
bundled by imageio-ffmpeg, then transcribed by whisper.cpp.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TextIO

from .models import TranscriptSegment


class LocalTranscriptionError(RuntimeError):
    """Raised when local audio preparation or transcription fails."""


SUPPORTED_AUDIO_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
}

# Multilingual models are used so that switches between English and Hebrew can
# remain visible. Quantized variants reduce download size and RAM usage.
MODEL_CHOICES = (
    "tiny-q5_1",
    "base-q5_1",
    "small-q5_1",
    "medium-q5_0",
    "large-v3-turbo-q5_0",
)
DEFAULT_MODEL = "small-q5_1"


def create_local_transcription(
    audio_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    language: str = "",
    threads: int | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[list[TranscriptSegment], dict[str, Any]]:
    """Transcribe an audio file locally and return timestamped segments.

    ``language`` may be blank or ``auto`` for detection. The selected model is
    downloaded by pywhispercpp on first use and cached for later sessions.

    The GUI is launched with ``pythonw.exe`` on Windows. In that mode Python can
    set ``sys.stdout`` and ``sys.stderr`` to ``None``. pywhispercpp uses tqdm
    while downloading a model, and tqdm requires a writable stream. The
    ``_writable_console_streams`` context supplies a temporary private stream
    for the complete model load and transcription operation.
    """

    source = Path(audio_path).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise LocalTranscriptionError(f"Audio file not found: {source}")
    if source.suffix.casefold() not in SUPPORTED_AUDIO_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_SUFFIXES))
        raise LocalTranscriptionError(f"Unsupported audio format '{source.suffix}'. Supported formats: {supported}.")
    if model_name not in MODEL_CHOICES:
        raise LocalTranscriptionError(f"Unsupported local model '{model_name}'.")

    thread_count = _normalise_thread_count(threads)
    language_code = language.strip().casefold()
    detect_language = not language_code or language_code == "auto"
    if detect_language:
        language_code = ""

    with tempfile.TemporaryDirectory(prefix="transcription_audio_") as temp_dir:
        prepared_audio = Path(temp_dir) / "audio_16khz_mono.wav"
        if status_callback:
            status_callback("Preparing a local 16 kHz mono audio copy...")
        _convert_audio(source, prepared_audio)

        if status_callback:
            status_callback(
                f"Loading local Whisper model '{model_name}'. The first use may download the model once."
            )

        # Keep fallback streams active while importing pywhispercpp, resolving or
        # downloading the model, running inference, and releasing the model.
        # This prevents tqdm from calling .write() on None under pythonw.exe.
        with _writable_console_streams():
            try:
                from pywhispercpp.model import Model
            except ImportError as exc:
                raise LocalTranscriptionError(
                    "The local transcription package is not installed. Close the application and run SETUP.bat again."
                ) from exc

            try:
                model = Model(
                    model_name,
                    n_threads=thread_count,
                    print_progress=False,
                    print_realtime=False,
                    print_timestamps=False,
                    translate=False,
                    no_context=True,
                    no_timestamps=False,
                    suppress_blank=True,
                    suppress_non_speech_tokens=False,
                    temperature=0.0,
                    language=language_code,
                    detect_language=detect_language,
                    redirect_whispercpp_logs_to=None,
                )
            except Exception as exc:
                raise LocalTranscriptionError(f"Could not load local Whisper model '{model_name}': {exc}") from exc

            segment_count = 0

            def on_segment(_segment: object) -> None:
                nonlocal segment_count
                segment_count += 1
                if status_callback:
                    status_callback(f"Transcribing locally: {segment_count} segments produced...")

            if status_callback:
                status_callback("Running local Whisper transcription. Audio remains on this computer...")
            try:
                raw_segments = model.transcribe(
                    str(prepared_audio),
                    new_segment_callback=on_segment,
                    extract_probability=True,
                )
            except Exception as exc:
                raise LocalTranscriptionError(f"Local Whisper transcription failed: {exc}") from exc
            finally:
                # Release the native model before the temporary stream closes.
                try:
                    del model
                except UnboundLocalError:
                    pass

    segments = _segments_from_whisper(raw_segments)
    if not segments:
        raise LocalTranscriptionError("The local model did not return any speech segments.")
    if status_callback:
        status_callback(f"Local transcription complete: {len(segments)} timestamped segments.")

    details = {
        "engine": "pywhispercpp",
        "backend": "whisper.cpp",
        "model": model_name,
        "language": language_code or "auto",
        "threads": thread_count,
        "local_processing": True,
    }
    return segments, details


@contextmanager
def _writable_console_streams() -> Iterator[None]:
    """Temporarily provide writable stdout/stderr streams when Python has none.

    Windows GUI programs started with ``pythonw.exe`` normally have no console,
    so ``sys.stdout`` and ``sys.stderr`` can be ``None``. Some dependencies still
    write progress information to those streams. A temporary text file is used
    only as a compatibility sink; it is automatically deleted on close.
    """

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    fallback: TextIO | None = None

    try:
        if not _is_writable_stream(original_stdout) or not _is_writable_stream(original_stderr):
            fallback = tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="")
            if not _is_writable_stream(original_stdout):
                sys.stdout = fallback
            if not _is_writable_stream(original_stderr):
                sys.stderr = fallback
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if fallback is not None:
            fallback.close()


def _is_writable_stream(stream: object) -> bool:
    if stream is None or bool(getattr(stream, "closed", False)):
        return False
    return callable(getattr(stream, "write", None)) and callable(getattr(stream, "flush", None))


def _convert_audio(source: Path, target: Path) -> None:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise LocalTranscriptionError(
            "The bundled audio-conversion package is not installed. Run SETUP.bat again."
        ) from exc

    try:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise LocalTranscriptionError(f"Could not locate the bundled FFmpeg executable: {exc}") from exc

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
        check=False,
    )
    if completed.returncode != 0 or not target.exists():
        error = completed.stderr.strip() or "Unknown FFmpeg error"
        raise LocalTranscriptionError(f"Could not prepare the audio file: {error}")


def _segments_from_whisper(raw_segments: Iterable[object]) -> list[TranscriptSegment]:
    """Convert pywhispercpp segments into project segments.

    whisper.cpp timestamps use 10-millisecond units, so ``t0 / 100`` and
    ``t1 / 100`` convert them to seconds.
    """

    converted: list[TranscriptSegment] = []
    for raw in raw_segments:
        text = str(getattr(raw, "text", "")).strip()
        if not text:
            continue
        try:
            start = float(getattr(raw, "t0")) / 100.0
            end = float(getattr(raw, "t1")) / 100.0
        except (TypeError, ValueError, AttributeError):
            start = None
            end = None
        probability = _finite_probability(getattr(raw, "probability", None))
        converted.append(
            TranscriptSegment(
                start=start,
                end=end,
                speaker="Unknown",
                text=text,
                confidence=probability,
            )
        )
    return converted


def _finite_probability(value: object) -> float | None:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(probability):
        return None
    return max(0.0, min(1.0, probability))


def _normalise_thread_count(value: int | None) -> int:
    available = max(1, os.cpu_count() or 1)
    if value is None:
        return min(8, available)
    try:
        requested = int(value)
    except (TypeError, ValueError):
        requested = min(8, available)
    return max(1, min(requested, available))
