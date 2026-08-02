"""Optional WAV-only signal features implemented with the Python standard library."""

from __future__ import annotations

import math
import wave
from array import array
from collections.abc import Iterable
from pathlib import Path


class AudioFeatureError(RuntimeError):
    pass


def _decode_samples(raw: bytes, sample_width: int) -> list[int]:
    if sample_width == 1:
        return [value - 128 for value in raw]
    if sample_width == 2:
        values = array("h")
        values.frombytes(raw)
        if values.itemsize != 2:
            raise AudioFeatureError("Unsupported platform integer layout.")
        return values.tolist()
    if sample_width == 3:
        output: list[int] = []
        for index in range(0, len(raw) - 2, 3):
            value = int.from_bytes(raw[index : index + 3], byteorder="little", signed=True)
            output.append(value)
        return output
    if sample_width == 4:
        values = array("i")
        values.frombytes(raw)
        return values.tolist()
    raise AudioFeatureError(f"Unsupported WAV sample width: {sample_width} bytes")


def _mono(samples: list[int], channels: int) -> list[float]:
    if channels <= 1:
        return [float(value) for value in samples]
    result: list[float] = []
    for index in range(0, len(samples) - channels + 1, channels):
        frame = samples[index : index + channels]
        result.append(sum(frame) / channels)
    return result


def _rms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _analyze_open_wav_interval(
    handle: wave.Wave_read,
    start: float | None,
    end: float | None,
) -> dict[str, float | None]:
    channels = handle.getnchannels()
    sample_width = handle.getsampwidth()
    rate = handle.getframerate()
    total_frames = handle.getnframes()
    start_frame = int(max(0.0, start or 0.0) * rate)
    end_seconds = end if end is not None else total_frames / rate
    end_frame = min(total_frames, int(max(start or 0.0, end_seconds) * rate))
    frame_count = max(0, end_frame - start_frame)
    if frame_count <= 0:
        return {"volume_dbfs": None, "noise_snr_db": None}
    handle.setpos(start_frame)
    raw = handle.readframes(frame_count)

    samples = _mono(_decode_samples(raw, sample_width), channels)
    if not samples:
        return {"volume_dbfs": None, "noise_snr_db": None}

    # Downsample for predictable performance on long turns.
    stride = max(1, len(samples) // 200_000)
    samples = samples[::stride]
    peak = float(2 ** (sample_width * 8 - 1))
    overall_rms = _rms(samples)
    volume_dbfs = 20.0 * math.log10(max(overall_rms, 1.0) / peak)

    block_size = max(64, min(len(samples), int(rate * 0.05 / stride)))
    block_rms = [
        _rms(samples[index : index + block_size])
        for index in range(0, len(samples), block_size)
    ]
    positive = sorted(value for value in block_rms if value > 0)
    if not positive:
        return {"volume_dbfs": volume_dbfs, "noise_snr_db": None}
    quiet_count = max(1, len(positive) // 10)
    noise_rms = sum(positive[:quiet_count]) / quiet_count
    snr_db = 20.0 * math.log10(max(overall_rms, 1.0) / max(noise_rms, 1.0))
    return {"volume_dbfs": volume_dbfs, "noise_snr_db": snr_db}


def analyze_wav_intervals(
    path: str | Path,
    intervals: Iterable[tuple[float | None, float | None]],
) -> list[dict[str, float | None]]:
    """Analyze multiple intervals while opening the WAV file only once."""
    source = Path(path)
    requested = list(intervals)
    if source.suffix.casefold() != ".wav":
        return [
            {"volume_dbfs": None, "noise_snr_db": None}
            for _ in requested
        ]
    try:
        with wave.open(str(source), "rb") as handle:
            return [
                _analyze_open_wav_interval(handle, start, end)
                for start, end in requested
            ]
    except (wave.Error, OSError) as exc:
        raise AudioFeatureError(str(exc)) from exc


def analyze_wav_interval(
    path: str | Path,
    start: float | None,
    end: float | None,
) -> dict[str, float | None]:
    return analyze_wav_intervals(path, [(start, end)])[0]
