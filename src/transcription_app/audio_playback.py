"""Turn-level audio preview using bundled FFmpeg and Windows winsound."""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from pathlib import Path


class TurnPlaybackError(RuntimeError):
    """Raised when a timestamped turn cannot be prepared or played."""


def normalize_playback_interval(
    start: float | None,
    end: float | None,
) -> tuple[float, float]:
    """Validate turn timestamps and return ``(start, duration)`` in seconds."""
    if start is None or end is None:
        raise TurnPlaybackError("This turn does not have usable start and end timestamps.")
    try:
        raw_start = float(start)
        raw_end = float(end)
    except (TypeError, ValueError) as exc:
        raise TurnPlaybackError("This turn has invalid audio timestamps.") from exc
    if not math.isfinite(raw_start) or not math.isfinite(raw_end):
        raise TurnPlaybackError("This turn has invalid audio timestamps.")
    start_seconds = max(0.0, raw_start)
    end_seconds = raw_end
    duration = end_seconds - start_seconds
    if duration <= 0.01:
        raise TurnPlaybackError("This turn has no playable audio duration.")
    return start_seconds, duration


def build_clip_command(
    ffmpeg: str,
    source: Path,
    target: Path,
    start: float,
    duration: float,
) -> list[str]:
    """Build the FFmpeg command that extracts one turn as PCM WAV."""
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(target),
    ]


class TurnAudioPlayer:
    """Extract and asynchronously play timestamped audio clips on Windows."""

    def __init__(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory(prefix="turn_audio_preview_")
        self._clip_number = 0
        self._closed = False

    def play(
        self,
        audio_path: str | Path,
        start: float | None,
        end: float | None,
    ) -> float:
        """Prepare and play a turn, returning its duration in seconds."""
        if self._closed:
            raise TurnPlaybackError("The audio preview player is no longer available.")
        if os.name != "nt":
            raise TurnPlaybackError("Turn audio preview is currently supported on Windows only.")

        source = Path(audio_path).expanduser().resolve()
        if not source.exists() or not source.is_file():
            raise TurnPlaybackError(f"Audio file not found: {source}")
        start_seconds, duration = normalize_playback_interval(start, end)

        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise TurnPlaybackError(
                "The bundled audio-conversion package is not installed. Run SETUP.bat again."
            ) from exc

        try:
            import winsound
        except ImportError as exc:
            raise TurnPlaybackError("Windows audio playback is unavailable in this Python installation.") from exc

        self.stop()
        self._clip_number += 1
        target = Path(self._temporary_directory.name) / f"turn_{self._clip_number:05d}.wav"
        try:
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            raise TurnPlaybackError(f"Could not locate the bundled FFmpeg executable: {exc}") from exc

        command = build_clip_command(
            ffmpeg,
            source,
            target,
            start_seconds,
            duration,
        )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
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
            detail = completed.stderr.strip() or "Unknown FFmpeg error"
            raise TurnPlaybackError(f"Could not prepare this turn for playback: {detail}")

        try:
            winsound.PlaySound(
                str(target),
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
            )
        except RuntimeError as exc:
            raise TurnPlaybackError(f"Could not play this turn: {exc}") from exc
        return duration

    @staticmethod
    def stop() -> None:
        """Stop asynchronous winsound playback when running on Windows."""
        if os.name != "nt":
            return
        try:
            import winsound

            winsound.PlaySound(None, 0)
        except (ImportError, RuntimeError):
            pass

    def close(self) -> None:
        """Stop playback and remove temporary preview clips."""
        if self._closed:
            return
        self.stop()
        try:
            self._temporary_directory.cleanup()
        except OSError:
            # Windows may briefly retain a handle after asynchronous playback.
            pass
        self._closed = True