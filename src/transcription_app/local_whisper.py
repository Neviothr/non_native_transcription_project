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
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TextIO

from .models import TranscriptSegment
from .speech_delays import (
    SpeechDelayConfig,
    SpeechDelayDetectionError,
    detect_speech_delays as _detect_speech_delays,
)


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

# Keep loaded native models alive for the lifetime of the application.
# Some Windows pywhispercpp builds can hang while destroying a Model after
# transcription. Reusing the model also avoids repeated loading.
PATCH_VERSION = "stage4-auto-language-fix-v3"
_MODEL_CACHE: dict[tuple[str, int, str, bool], Any] = {}
_CONSOLE_SINK: TextIO | None = None


def available_cpu_threads() -> int:
    """Return the maximum CPU thread count exposed by the operating system."""

    return max(1, os.cpu_count() or 1)


def create_local_transcription(
    audio_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    language: str = "",
    threads: int | None = None,
    status_callback: Callable[[str], None] | None = None,
    detect_speech_delays: bool = True,
    minimum_pause_seconds: float = 0.3,
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
    ``_writable_console_streams`` supplies a process-lifetime null stream
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
    source_stat = source.stat()
    audio_source_path = str(source)
    audio_source_size_bytes = source_stat.st_size
    audio_source_modified_time_ns = source_stat.st_mtime_ns

    thread_count = _normalise_thread_count(threads)
    requested_language = language.strip().casefold()
    automatic_language = not requested_language or requested_language == "auto"
    # For automatic transcription, whisper.cpp expects language to be empty/auto.
    # Setting detect_language=True can produce a detection-only pass in the
    # affected Windows build, returning no transcript segments.
    language_code = "auto" if automatic_language else requested_language
    detect_language = False

    _emit_status(
        status_callback,
        f"Patch active: {PATCH_VERSION}. "
        "Automatic language selection uses language=auto with detect_language=False; "
        "the Whisper model is retained for reuse.",
    )

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
    speech_delay_detection_seconds = 0.0
    speech_delay_events: list[dict[str, Any]] = []
    speech_delay_detection_error = ""

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

        if detect_speech_delays:
            _emit_status(
                status_callback,
                "Stage 1/5 - Non-destructive silent-pause analysis started; "
                f"minimum duration={minimum_pause_seconds:.2f} seconds.",
            )
            delay_started = time.monotonic()
            try:
                detected_delays = _detect_speech_delays(
                    prepared_audio,
                    SpeechDelayConfig(
                        minimum_pause_seconds=float(minimum_pause_seconds),
                    ),
                )
                speech_delay_events = [
                    {
                        **asdict(item),
                        "audio_source_path": audio_source_path,
                        "audio_source_size_bytes": audio_source_size_bytes,
                        "audio_source_modified_time_ns": audio_source_modified_time_ns,
                    }
                    for item in detected_delays
                ]
            except (SpeechDelayDetectionError, TypeError, ValueError) as exc:
                # Pause analysis is supplementary evidence. A detector failure
                # must not discard an otherwise valid local transcription.
                speech_delay_detection_error = str(exc)
                _emit_status(
                    status_callback,
                    "Stage 1/5 - WARNING: silent-pause analysis was unavailable: "
                    f"{exc}",
                )
            speech_delay_detection_seconds = time.monotonic() - delay_started
            _emit_status(
                status_callback,
                "Stage 1/5 - Silent-pause analysis finished in "
                f"{_format_duration(speech_delay_detection_seconds)}; retained "
                f"{len(speech_delay_events)} timed candidate(s).",
            )
        else:
            _emit_status(
                status_callback,
                "Stage 1/5 - Silent-pause analysis disabled for this run.",
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

            model_key = (model_name, thread_count, language_code, detect_language)
            model = _MODEL_CACHE.get(model_key)
            if model is None:
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
                _MODEL_CACHE[model_key] = model
                model_state = "loaded and retained for reuse"
            else:
                model_state = "reused from the in-memory model cache"

            model_load_seconds = time.monotonic() - model_load_started
            _emit_status(
                status_callback,
                "Stage 2/5 - Model ready in "
                f"{_format_duration(model_load_seconds)}; {model_state}.",
            )

            # Copy each native pywhispercpp segment immediately inside the
            # callback. Some Windows/Python builds can deadlock when the same
            # native-backed segment objects are read again after transcribe()
            # returns. Keeping only ordinary Python dataclasses removes that
            # second native-object access from Stage 4.
            segment_count = 0
            callback_segments: list[TranscriptSegment] = []

            def on_segment(segment: object) -> None:
                nonlocal segment_count
                segment_count += 1

                copied_segment = _segment_from_whisper(segment)
                if copied_segment is not None:
                    callback_segments.append(copied_segment)

                _emit_status(
                    status_callback,
                    _segment_log_message_from_copy(
                        copied_segment,
                        segment_count,
                    ),
                )

            _emit_status(
                status_callback,
                "Stage 3/5 - Whisper inference started. "
                f"Audio duration={_format_duration(audio_duration_seconds)}; "
                f"language={'automatic detection' if automatic_language else language_code}; "
                f"threads={thread_count}.",
            )
            inference_started = time.monotonic()
            try:
                model.transcribe(
                    str(prepared_audio),
                    new_segment_callback=on_segment,
                    extract_probability=True,
                )

                inference_seconds = time.monotonic() - inference_started
                realtime_factor = (
                    inference_seconds / audio_duration_seconds
                    if audio_duration_seconds > 0
                    else 0.0
                )
                _emit_status(
                    status_callback,
                    "Stage 3/5 - Whisper inference finished in "
                    f"{_format_duration(inference_seconds)}. Captured "
                    f"{len(callback_segments)} independent callback segments from "
                    f"{segment_count} callback events; "
                    f"real-time factor={realtime_factor:.3f}x.",
                )

                _emit_status(
                    status_callback,
                    "Stage 4/5 - Finalizing the independent callback copies. "
                    "No native Whisper segment objects will be re-read.",
                )
                postprocess_started = time.monotonic()

                segments = callback_segments
                segment_source = "callback copies"

                postprocess_seconds = time.monotonic() - postprocess_started
                if not segments:
                    raise LocalTranscriptionError(
                        "Whisper returned zero speech segments. With language=auto this "
                        "usually means the recording contains no detectable speech, the "
                        "audio is extremely quiet, or the selected model file is invalid."
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
                    f"{_format_duration(postprocess_seconds)} using {segment_source}: "
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
                # Do not destroy the native Model here. Some Windows builds can
                # hang in the native destructor after successful inference. The
                # module-level cache intentionally retains it until the process
                # exits, allowing Stage 5 and the GUI workflow to continue.
                _emit_status(
                    status_callback,
                    "Stage 4/5 - Native Whisper model retained in memory; "
                    "skipping the known-hanging model teardown step.",
                )

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
        "audio_source_path": audio_source_path,
        "audio_source_size_bytes": audio_source_size_bytes,
        "audio_source_modified_time_ns": audio_source_modified_time_ns,
        "conversion_seconds": conversion_seconds,
        "speech_delay_detection_enabled": bool(detect_speech_delays),
        "minimum_pause_seconds": float(minimum_pause_seconds),
        "speech_delay_detection_seconds": speech_delay_detection_seconds,
        "speech_delay_detection_error": speech_delay_detection_error,
        "speech_delay_events": speech_delay_events,
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
    """Format a native segment safely while it is still inside the callback."""
    return _segment_log_message_from_copy(_segment_from_whisper(segment), index)


def _segment_log_message_from_copy(
    segment: TranscriptSegment | None,
    index: int,
) -> str:
    if segment is None:
        return (
            f"Stage 3/5 - Segment {index:04d} "
            "[timestamp unavailable]: [empty text returned by callback]"
        )

    text = " ".join(segment.text.split())
    if len(text) > 180:
        text = text[:177] + "..."

    if segment.start is not None and segment.end is not None:
        time_range = (
            f"{_format_duration(segment.start)} -> "
            f"{_format_duration(segment.end)}"
        )
    else:
        time_range = "timestamp unavailable"

    return (
        f"Stage 3/5 - Segment {index:04d} [{time_range}]: "
        f"{text or '[empty text returned by callback]'}"
    )


@contextmanager
def _writable_console_streams() -> Iterator[None]:
    """Provide writable stdout/stderr under pythonw without per-run teardown.

    A process-lifetime null sink is used when Windows launches the application
    without console streams. It is intentionally not closed after each run:
    dependencies may retain a reference to the stream, and closing it during an
    exception path can prevent the GUI from receiving the real transcription
    error.
    """

    global _CONSOLE_SINK

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        needs_stdout = not _is_writable_stream(original_stdout)
        needs_stderr = not _is_writable_stream(original_stderr)
        if needs_stdout or needs_stderr:
            if _CONSOLE_SINK is None or bool(getattr(_CONSOLE_SINK, "closed", False)):
                _CONSOLE_SINK = open(
                    os.devnull,
                    mode="w",
                    encoding="utf-8",
                    errors="replace",
                )
            if needs_stdout:
                sys.stdout = _CONSOLE_SINK
            if needs_stderr:
                sys.stderr = _CONSOLE_SINK
        yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


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


def _segment_from_whisper(raw: object) -> TranscriptSegment | None:
    """Copy one native pywhispercpp segment into ordinary Python data.

    whisper.cpp timestamps use 10-millisecond units, so ``t0 / 100`` and
    ``t1 / 100`` convert them to seconds. This function should preferably be
    called from ``new_segment_callback`` while the native segment is valid.
    """

    text = str(getattr(raw, "text", "")).strip()
    if not text:
        return None

    try:
        start = float(getattr(raw, "t0")) / 100.0
        end = float(getattr(raw, "t1")) / 100.0
    except (TypeError, ValueError, AttributeError):
        start = None
        end = None

    probability = _finite_probability(getattr(raw, "probability", None))
    return TranscriptSegment(
        start=start,
        end=end,
        speaker="Unknown",
        text=text,
        confidence=probability,
    )


def _segments_from_whisper(raw_segments: Iterable[object]) -> list[TranscriptSegment]:
    """Convert an iterable of pywhispercpp segments into project segments."""

    converted: list[TranscriptSegment] = []
    for raw in raw_segments:
        segment = _segment_from_whisper(raw)
        if segment is not None:
            converted.append(segment)
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
    available = available_cpu_threads()
    if value is None:
        return available
    try:
        requested = int(value)
    except (TypeError, ValueError):
        requested = available
    return max(1, min(requested, available))
