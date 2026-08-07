"""Non-destructive silent-pause detection for prepared mono PCM WAV audio.

The detector reports acoustic evidence only.  It never edits transcript text,
cuts audio, or compresses the source timeline.  Returned timestamps are always
absolute offsets into the source WAV file, even when only a sub-interval is
analysed.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


class SpeechDelayDetectionError(RuntimeError):
    """Raised when a WAV file cannot be analysed as mono PCM audio."""


@dataclass(frozen=True, slots=True)
class SpeechDelayConfig:
    """Configuration for frame-based silent-pause detection.

    ``edge_padding_seconds`` is an interval-edge guard, not audio padding.
    Unless ``include_interval_edges`` is enabled, a silent run is ignored when
    it begins or ends within this distance of the requested interval boundary.
    This keeps leading/trailing turn silence separate from an internal delay.
    """

    silence_threshold_dbfs: float = -40.0
    minimum_pause_seconds: float = 0.30
    frame_seconds: float = 0.02
    edge_padding_seconds: float = 0.05
    include_interval_edges: bool = False

    def __post_init__(self) -> None:
        numeric_fields = {
            "silence_threshold_dbfs": self.silence_threshold_dbfs,
            "minimum_pause_seconds": self.minimum_pause_seconds,
            "frame_seconds": self.frame_seconds,
            "edge_padding_seconds": self.edge_padding_seconds,
        }
        for name, value in numeric_fields.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if self.silence_threshold_dbfs > 0.0:
            raise ValueError("silence_threshold_dbfs cannot exceed 0 dBFS.")
        if self.minimum_pause_seconds <= 0.0:
            raise ValueError("minimum_pause_seconds must be greater than zero.")
        if self.frame_seconds <= 0.0:
            raise ValueError("frame_seconds must be greater than zero.")
        if self.edge_padding_seconds < 0.0:
            raise ValueError("edge_padding_seconds cannot be negative.")


@dataclass(frozen=True, slots=True)
class DetectedDelay:
    """One raw silent-pause observation on the source audio timeline."""

    interval_index: int
    interval_start_seconds: float
    interval_end_seconds: float
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    loudest_frame_dbfs: float | None
    event_type: str = field(default="silent_pause", init=False)


@dataclass(frozen=True, slots=True)
class _AnalysisFrame:
    start_frame: int
    end_frame: int
    dbfs: float | None
    silent: bool


def _sample_value(raw: bytes, offset: int, sample_width: int) -> int:
    sample = raw[offset : offset + sample_width]
    if sample_width == 1:
        return sample[0] - 128
    return int.from_bytes(sample, byteorder="little", signed=True)


def _frame_dbfs(raw: bytes, sample_width: int) -> float | None:
    sample_count = len(raw) // sample_width
    if sample_count <= 0:
        return None

    square_total = 0.0
    for offset in range(0, sample_count * sample_width, sample_width):
        value = _sample_value(raw, offset, sample_width)
        square_total += float(value * value)
    rms = math.sqrt(square_total / sample_count)
    if rms <= 0.0:
        return None

    full_scale = float(2 ** (sample_width * 8 - 1))
    return 20.0 * math.log10(rms / full_scale)


def _finite_seconds(value: float | None, *, name: str, default: float) -> float:
    if value is None:
        return default
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number or None.") from exc
    if not math.isfinite(seconds):
        raise ValueError(f"{name} must be finite.")
    return seconds


def _normalise_interval(
    start: float | None,
    end: float | None,
    *,
    duration_seconds: float,
    sample_rate: int,
) -> tuple[float, float, int, int]:
    requested_start = _finite_seconds(start, name="interval start", default=0.0)
    requested_end = _finite_seconds(
        end,
        name="interval end",
        default=duration_seconds,
    )
    interval_start = min(duration_seconds, max(0.0, requested_start))
    interval_end = min(duration_seconds, max(0.0, requested_end))
    if interval_end <= interval_start:
        return interval_start, interval_start, 0, 0

    # Analyse only samples that lie inside the requested interval.  Ceil at the
    # left edge and floor at the right edge avoid leaking adjacent-turn audio.
    start_frame = math.ceil(interval_start * sample_rate)
    end_frame = math.floor(interval_end * sample_rate)
    if end_frame <= start_frame:
        return interval_start, interval_end, 0, 0
    return interval_start, interval_end, start_frame, end_frame


def _read_analysis_frames(
    handle: wave.Wave_read,
    *,
    start_frame: int,
    end_frame: int,
    sample_width: int,
    samples_per_frame: int,
    silence_threshold_dbfs: float,
) -> list[_AnalysisFrame]:
    handle.setpos(start_frame)
    remaining = end_frame - start_frame
    cursor = start_frame
    frames: list[_AnalysisFrame] = []

    while remaining > 0:
        requested = min(samples_per_frame, remaining)
        raw = handle.readframes(requested)
        actual = len(raw) // sample_width
        if actual <= 0:
            break
        dbfs = _frame_dbfs(raw, sample_width)
        frames.append(
            _AnalysisFrame(
                start_frame=cursor,
                end_frame=cursor + actual,
                dbfs=dbfs,
                silent=dbfs is None or dbfs <= silence_threshold_dbfs,
            )
        )
        cursor += actual
        remaining -= actual
    return frames


def _silent_runs(frames: list[_AnalysisFrame]) -> list[list[_AnalysisFrame]]:
    runs: list[list[_AnalysisFrame]] = []
    current: list[_AnalysisFrame] = []
    for frame in frames:
        if frame.silent:
            current.append(frame)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _detect_in_interval(
    handle: wave.Wave_read,
    *,
    interval_index: int,
    interval_start: float,
    interval_end: float,
    start_frame: int,
    end_frame: int,
    sample_width: int,
    sample_rate: int,
    config: SpeechDelayConfig,
) -> list[DetectedDelay]:
    if end_frame <= start_frame:
        return []

    samples_per_frame = max(1, round(config.frame_seconds * sample_rate))
    frames = _read_analysis_frames(
        handle,
        start_frame=start_frame,
        end_frame=end_frame,
        sample_width=sample_width,
        samples_per_frame=samples_per_frame,
        silence_threshold_dbfs=config.silence_threshold_dbfs,
    )

    detections: list[DetectedDelay] = []
    for run in _silent_runs(frames):
        run_start = run[0].start_frame / sample_rate
        run_end = run[-1].end_frame / sample_rate
        duration = run_end - run_start
        if duration + (0.5 / sample_rate) < config.minimum_pause_seconds:
            continue

        if not config.include_interval_edges:
            left_guard = interval_start + config.edge_padding_seconds
            right_guard = interval_end - config.edge_padding_seconds
            if run_start <= left_guard or run_end >= right_guard:
                continue

        finite_levels = [frame.dbfs for frame in run if frame.dbfs is not None]
        detections.append(
            DetectedDelay(
                interval_index=interval_index,
                interval_start_seconds=interval_start,
                interval_end_seconds=interval_end,
                start_seconds=run_start,
                end_seconds=run_end,
                duration_seconds=duration,
                loudest_frame_dbfs=max(finite_levels) if finite_levels else None,
            )
        )
    return detections


def detect_speech_delays(
    path: str | Path,
    config: SpeechDelayConfig | None = None,
    *,
    intervals: Iterable[tuple[float | None, float | None]] | None = None,
) -> list[DetectedDelay]:
    """Detect internal silent pauses in a WAV file or selected intervals.

    With ``intervals=None``, the whole file is analysed as interval zero.  The
    WAV is opened once and read only.  Results from explicitly supplied
    intervals are flattened and retain ``interval_index`` for association with
    the caller's turn list.  Overlapping input intervals are analysed
    independently and can therefore yield overlapping duplicate observations.
    """

    requested = [(None, None)] if intervals is None else list(intervals)
    if intervals is not None and not requested:
        return []
    settings = config or SpeechDelayConfig()
    source = Path(path)

    try:
        with wave.open(str(source), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            total_frames = handle.getnframes()
            compression = handle.getcomptype()

            if compression != "NONE":
                raise SpeechDelayDetectionError(
                    f"Unsupported compressed WAV format: {compression}."
                )
            if channels != 1:
                raise SpeechDelayDetectionError(
                    f"Expected mono WAV audio, found {channels} channels."
                )
            if sample_width not in {1, 2, 3, 4}:
                raise SpeechDelayDetectionError(
                    f"Unsupported PCM sample width: {sample_width} bytes."
                )
            if sample_rate <= 0:
                raise SpeechDelayDetectionError("WAV sample rate must be positive.")

            duration_seconds = total_frames / sample_rate
            detections: list[DetectedDelay] = []
            for interval_index, (start, end) in enumerate(requested):
                (
                    interval_start,
                    interval_end,
                    start_frame,
                    end_frame,
                ) = _normalise_interval(
                    start,
                    end,
                    duration_seconds=duration_seconds,
                    sample_rate=sample_rate,
                )
                detections.extend(
                    _detect_in_interval(
                        handle,
                        interval_index=interval_index,
                        interval_start=interval_start,
                        interval_end=interval_end,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        sample_width=sample_width,
                        sample_rate=sample_rate,
                        config=settings,
                    )
                )
            return detections
    except SpeechDelayDetectionError:
        raise
    except (OSError, wave.Error) as exc:
        raise SpeechDelayDetectionError(str(exc)) from exc


def detect_speech_delays_in_interval(
    path: str | Path,
    start: float | None,
    end: float | None,
    *,
    config: SpeechDelayConfig | None = None,
) -> list[DetectedDelay]:
    """Convenience wrapper for analysing a single source interval."""

    return detect_speech_delays(path, config, intervals=[(start, end)])
