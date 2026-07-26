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
import time
import wave
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

    The status callback intentionally reports detailed stage, timing, file, and
    segment information so the GUI log can be used to diagnose slow or failed
    runs without opening a command prompt.

    The GUI is launched with ``pythonw.exe`` on Windows. In that mode Python can
    set ``sys.stdout`` and ``sys.stderr`` to ``None``. pywhispercpp uses tqdm
    while downloading a model, and tqdm requires a writable stream. The
    ``_writable_console_streams`` context supplies a temporary private stream
    for the complete model load and transcription operation.
    """

    run_started = time.monotonic()
    source = Path(audio_path).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise LocalTranscriptionError(f"Audio file not found: {source}")
    if source.suffix.casefold() not in SUPPORTED_AUDIO_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_SUFFIXES))
        raise LocalTranscriptionError(
            f"Unsupported audio format '{source.suffix}'. Supported formats: {supported}."
        )
    if model_name not in MODEL_CHOICES:
        raise LocalTranscriptionError(f"Unsupported local model '{model_name}'.")

    thread_count = _normalise_thread_count(threads)
    language_code = language.strip().casefold()
    detect_language = not language_code or language_code == "auto"
    if detect_language:
        language_code = ""

    _emit_status(
        status_callback,
        "Validation complete: "
        f"{source.name} ({_format_file_size(source.stat().st_size)}), "
        f"model={model_name}, language={language_code or 'auto'}, threads={thread_count}.",
    )

    conversion_seconds = 0.0
    model_load_seconds = 0.0
    inference_seconds = 0.0
    audio_duration_seconds = 0.0

    with tempfile.TemporaryDirectory(prefix="transcription_audio_") as temp_dir:
        prepared_audio = Path(temp_dir) / "audio_16khz_mono.wav"
        _emit_status(
            status_callback,
            "Stage 1/5 - Audio preparation started: converting to 16 kHz, mono, "
            "16-bit PCM WAV for Whisper.",
        )
        conversion_started = time.monotonic()
        _convert_audio(source, prepared_audio)
        conversion_seconds = time.monotonic() - conversion_started
        audio_duration_seconds = _wav_duration_seconds(prepared_audio)
        _emit_status(
            status_callback,
            "Stage 1/5 - Audio preparation complete in "
            f"{_format_duration(conversion_seconds)}. Prepared file: "
            f"{_format_file_size(prepared_audio.stat().st_size)}; duration: "
            f"{_format_duration(audio_duration_seconds)}.",
        )

        _emit_status(
            status_callback,
            f"Stage 2/5 - Loading Whisper model '{model_name}'. "
            "A first-time model download can make this stage much longer.",
        )
        model_load_started = time.monotonic()

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
                raise LocalTranscriptionError(
                    f"Could not load local Whisper model '{model_name}': {exc}"
                ) from exc

            model_load_seconds = time.monotonic() - model_load_started
            _emit_status(
                status_callback,
                f"Stage 2/5 - Model ready in {_format_duration(model_load_seconds)}.",
            )

            segment_count = 0

            def on_segment(segment: object) -> None:
                nonlocal segment_count
                segment_count += 1
                _emit_status(
                    status_callback,
                    _segment_log_message(segment, segment_count),
                )

            _emit_status(
                status_callback,
                "Stage 3/5 - Whisper inference started. "
                f"Audio duration={_format_duration(audio_duration_seconds)}; "
                f"language={'automatic detection' if detect_language else language_code}; "
                f"threads={thread_count}.",
            )
            inference_started = time.monotonic()
            try:
                returned_segments = model.transcribe(
                    str(prepared_audio),
                    new_segment_callback=on_segment,
                    extract_probability=True,
                )

                # pywhispercpp documents the return value as an iterable of
                # segments. Materialize it while the native model and prepared
                # audio are still alive. Some versions/builds may return a lazy
                # or native-backed iterable that cannot be consumed safely after
                # the model has been released.
                raw_segments = list(returned_segments)
                inference_seconds = time.monotonic() - inference_started
                realtime_factor = (
                    inference_seconds / audio_duration_seconds
                    if audio_duration_seconds > 0
                    else 0.0
                )
                _emit_status(
                    status_callback,
                    "Stage 3/5 - Whisper inference and segment collection finished in "
                    f"{_format_duration(inference_seconds)}. Collected "
                    f"{len(raw_segments)} returned segment objects and observed "
                    f"{segment_count} callback segments; "
                    f"real-time factor={realtime_factor:.3f}x.",
                )

                _emit_status(
                    status_callback,
                    "Stage 4/5 - Converting the collected Whisper output into "
                    "independent project transcript segments while the model is still loaded.",
                )
                postprocess_started = time.monotonic()
                segments = _segments_from_whisper(raw_segments)
                postprocess_seconds = time.monotonic() - postprocess_started
                if not segments:
                    raise LocalTranscriptionError(
                        "The local model did not return any non-empty speech segments."
                    )

                first_start = next(
                    (segment.start for segment in segments if segment.start is not None),
                    None,
                )
                final_end = next(
                    (segment.end for segment in reversed(segments) if segment.end is not None),
                    None,
                )
                covered_duration = (
                    max(0.0, final_end - first_start)
                    if first_start is not None and final_end is not None
                    else 0.0
                )
                _emit_status(
                    status_callback,
                    "Stage 4/5 - Post-processing complete in "
                    f"{_format_duration(postprocess_seconds)}: "
                    f"{len(segments)} non-empty timestamped segments covering "
                    f"approximately {_format_duration(covered_duration)}.",
                )
            except LocalTranscriptionError:
                raise
            except Exception as exc:
                raise LocalTranscriptionError(
                    "Local Whisper transcription or segment collection failed: "
                    f"{exc}"
                ) from exc
            finally:
                # Segment output has now been copied into ordinary project data,
                # so the native model can be released safely.
                try:
                    del model
                except UnboundLocalError:
                    pass

    _emit_status(
        status_callback,
        "Stage 5/5 - Temporary 16 kHz audio and working files were removed.",
    )
    local_transcription_seconds = time.monotonic() - run_started
    _emit_status(
        status_callback,
        "Local Whisper stage complete in "
        f"{_format_duration(local_transcription_seconds)}. The GUI will now align "
        "sources and calculate initial quality flags.",
    )

    details = {
        "engine": "pywhispercpp",
        "backend": "whisper.cpp",
        "model": model_name,
        "language": language_code or "auto",
        "threads": thread_count,
        "local_processing": True,
        "audio_duration_seconds": audio_duration_seconds,
        "conversion_seconds": conversion_seconds,
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "local_transcription_seconds": local_transcription_seconds,
    }
    return segments, details


def _emit_status(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _format_duration(seconds: float) -> str:
    total_hundredths = max(0, int(seconds * 100))
    whole_seconds, hundredths = divmod(total_hundredths, 100)
    minutes, second = divmod(whole_seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{second:02d}.{hundredths:02d}"


def _format_file_size(size: int) -> str:
    value = float(max(0, size))
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _wav_duration_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            frame_rate = handle.getframerate()
            return handle.getnframes() / frame_rate if frame_rate > 0 else 0.0
    except (OSError, wave.Error):
        return 0.0


def _segment_log_message(segment: object, index: int) -> str:
    text = " ".join(str(getattr(segment, "text", "")).split())
    if len(text) > 180:
        text = text[:177] + "..."
    try:
        start = float(getattr(segment, "t0")) / 100.0
        end = float(getattr(segment, "t1")) / 100.0
        time_range = f"{_format_duration(start)} -> {_format_duration(end)}"
    except (TypeError, ValueError, AttributeError):
        time_range = "timestamp unavailable"
    return (
        f"Stage 3/5 - Segment {index:04d} [{time_range}]: "
        f"{text or '[empty text returned by callback]'}"
    )


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
