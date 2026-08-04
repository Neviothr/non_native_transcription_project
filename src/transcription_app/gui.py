"""Tkinter desktop interface for the transcription review workbench."""

from __future__ import annotations

import json
import os
import queue
import re
import textwrap
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from collections.abc import Callable
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import __last_revision_date__, __version__
from .audio_playback import TurnAudioPlayer, TurnPlaybackError
from .evaluation import evaluate_turns, per_source_metrics
from .models import ProjectData, ProjectMetadata, Turn
from .quality import FEATURE_NAMES
from .local_whisper import (
    DEFAULT_MODEL,
    MODEL_CHOICES,
    SUPPORTED_AUDIO_SUFFIXES,
    available_cpu_threads,
    create_local_transcription,
)
from .reporting import export_html_report
from .storage import load_project, save_project
from .workflow import (
    QUALITY_LABEL_TARGET,
    QUALITY_TRAINING_SCHEMA_VERSION,
    align_all_sources,
    analyze_turns,
    automatically_map_speakers,
    append_training_examples,
    import_source,
    initialize_turns_from_model,
    load_quality_model_if_available,
    normalize_role_for_conversation_type,
    normalize_speaker_identity,
    propagate_detected_learner_identity,
    reload_selected_transcripts,
    recover_speaker_mapping,
    resolve_turn_speaker_identity,
    speaker_label_for_turn,
    speaker_roles_for_conversation_type,
    train_quality_model,
)
from .xlsx_writer import export_xlsx
from .tooltips import install_button_tooltips


APP_TITLE = "Transcription Review Workbench"
APP_WINDOW_TITLE = f"{APP_TITLE} v{__version__}"
APP_RELEASE_LABEL = (
    f"Version {__version__}  |  Last revision: {__last_revision_date__}"
)
MAX_REVIEW_TREE_LINES = 12

REVIEW_TURN_COLUMNS = (
    "turn",
    "time",
    "speaker",
    "quality",
    "listen",
    "text",
)

AUDIO_SUFFIXES = tuple(sorted(SUPPORTED_AUDIO_SUFFIXES))
REVIEW_AUTOSAVE_DELAY_MS = 1_500
PROGRESS_UPDATE_INTERVAL_SECONDS = 0.1
_STAGE_PROGRESS_RE = re.compile(r"\bStage\s+(\d+)/(\d+)\b", re.IGNORECASE)
_ITEM_PROGRESS_RE = re.compile(r"\bturn\s+(\d+)\s+of\s+(\d+)\b", re.IGNORECASE)
_DIFF_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
STANDARD_TRANSCRIPT_SUFFIXES = (".vtt", ".srt", ".txt", ".csv", ".tsv", ".md")
XLSX_TRANSCRIPT_SUFFIXES = STANDARD_TRANSCRIPT_SUFFIXES + (".xlsx",)
TRANSCRIPT_SUFFIXES_BY_SOURCE = {
    "zoom": STANDARD_TRANSCRIPT_SUFFIXES,
    "chatgpt": XLSX_TRANSCRIPT_SUFFIXES,
    "gold": XLSX_TRANSCRIPT_SUFFIXES,
}

AUDIO_FILTERS = [
    ("Audio files", " ".join(f"*{suffix}" for suffix in AUDIO_SUFFIXES)),
    ("All files", "*.*"),
]
TRANSCRIPT_FILTERS_BY_SOURCE = {
    source_name: [
        ("Transcript files", " ".join(f"*{suffix}" for suffix in suffixes)),
        ("All files", "*.*"),
    ]
    for source_name, suffixes in TRANSCRIPT_SUFFIXES_BY_SOURCE.items()
}

AUDIO_FORMAT_NOTE = "Supported: " + ", ".join(
    suffix.removeprefix(".").upper() for suffix in AUDIO_SUFFIXES
)
TRANSCRIPT_FORMAT_NOTES_BY_SOURCE = {
    source_name: "Supported: " + ", ".join(
        suffix.removeprefix(".").upper() for suffix in suffixes
    )
    for source_name, suffixes in TRANSCRIPT_SUFFIXES_BY_SOURCE.items()
}
LAST_SELECTED_FILES_NAME = "last_selected_files.json"
LAST_SELECTED_FILE_KEYS = ("audio", "zoom", "chatgpt", "gold")

WHISPER_LANGUAGE_CODES = (
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
    "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
    "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
    "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
    "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
    "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
    "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
    "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
)
LARGE_V3_EXTRA_LANGUAGE_CODES = ("yue",)

WHISPER_LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French",
    "ja": "Japanese",
    "pt": "Portuguese",
    "tr": "Turkish",
    "pl": "Polish",
    "ca": "Catalan",
    "nl": "Dutch",
    "ar": "Arabic",
    "sv": "Swedish",
    "it": "Italian",
    "id": "Indonesian",
    "hi": "Hindi",
    "fi": "Finnish",
    "vi": "Vietnamese",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "el": "Greek",
    "ms": "Malay",
    "cs": "Czech",
    "ro": "Romanian",
    "da": "Danish",
    "hu": "Hungarian",
    "ta": "Tamil",
    "no": "Norwegian",
    "th": "Thai",
    "ur": "Urdu",
    "hr": "Croatian",
    "bg": "Bulgarian",
    "lt": "Lithuanian",
    "la": "Latin",
    "mi": "Māori",
    "ml": "Malayalam",
    "cy": "Welsh",
    "sk": "Slovak",
    "te": "Telugu",
    "fa": "Persian",
    "lv": "Latvian",
    "bn": "Bengali",
    "sr": "Serbian",
    "az": "Azerbaijani",
    "sl": "Slovenian",
    "kn": "Kannada",
    "et": "Estonian",
    "mk": "Macedonian",
    "br": "Breton",
    "eu": "Basque",
    "is": "Icelandic",
    "hy": "Armenian",
    "ne": "Nepali",
    "mn": "Mongolian",
    "bs": "Bosnian",
    "kk": "Kazakh",
    "sq": "Albanian",
    "sw": "Swahili",
    "gl": "Galician",
    "mr": "Marathi",
    "pa": "Punjabi",
    "si": "Sinhala",
    "km": "Khmer",
    "sn": "Shona",
    "yo": "Yoruba",
    "so": "Somali",
    "af": "Afrikaans",
    "oc": "Occitan",
    "ka": "Georgian",
    "be": "Belarusian",
    "tg": "Tajik",
    "sd": "Sindhi",
    "gu": "Gujarati",
    "am": "Amharic",
    "yi": "Yiddish",
    "lo": "Lao",
    "uz": "Uzbek",
    "fo": "Faroese",
    "ht": "Haitian Creole",
    "ps": "Pashto",
    "tk": "Turkmen",
    "nn": "Nynorsk",
    "mt": "Maltese",
    "sa": "Sanskrit",
    "lb": "Luxembourgish",
    "my": "Myanmar",
    "bo": "Tibetan",
    "tl": "Tagalog",
    "mg": "Malagasy",
    "as": "Assamese",
    "tt": "Tatar",
    "haw": "Hawaiian",
    "ln": "Lingala",
    "ha": "Hausa",
    "ba": "Bashkir",
    "jw": "Javanese",
    "su": "Sundanese",
    "yue": "Cantonese",
}

ALL_WHISPER_LANGUAGE_CODES = (
    *WHISPER_LANGUAGE_CODES,
    *LARGE_V3_EXTRA_LANGUAGE_CODES,
)
LANGUAGE_CHOICES = (
    "auto",
    *(f"{code} ({WHISPER_LANGUAGE_NAMES[code]})" for code in ALL_WHISPER_LANGUAGE_CODES),
)
LANGUAGE_CODE_NOTE = (
    "Select auto for automatic language detection. "
    "Cantonese (yue) is available only with large-v3-turbo-q5_0."
)


def _language_code_from_choice(value: str) -> str:
    """Return the Whisper code from a displayed language dropdown value."""
    selected = value.strip()
    if not selected or selected.casefold() == "auto":
        return "auto"
    return selected.split(maxsplit=1)[0].casefold()


def _last_selected_files_path() -> Path:
    """Return the per-user path used for unsaved Project Inputs choices."""
    local_data = os.environ.get("LOCALAPPDATA")
    base = Path(local_data) if local_data else Path.home() / ".config"
    return base / "Transcription Review Workbench" / LAST_SELECTED_FILES_NAME


def _load_last_selected_files(path: str | Path) -> dict[str, str]:
    """Load remembered input paths, ignoring missing or malformed preferences."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        key: value
        for key in LAST_SELECTED_FILE_KEYS
        if isinstance((value := raw.get(key)), str)
    }


def _save_last_selected_files(path: str | Path, files: dict[str, str]) -> None:
    """Persist the latest Project Inputs paths without requiring a project save."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: str(files.get(key, "")) for key in LAST_SELECTED_FILE_KEYS}
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")



BUTTON_TOOLTIPS = {
    "Browse...": "Opens a file-selection window for the file requested on this row.",
    "Save Project": "Saves the current project data, review edits, settings, and file references.",
    "Continue to Transcription": "Opens the Transcribe tab to configure and run the local Whisper model.",
    "Run Local Transcription": (
        "Reloads every selected transcript from disk, converts the selected audio "
        "locally, runs the chosen Whisper model, aligns sources, and creates review turns."
    ),
    "Open Review": "Opens the Review Turns tab.",
    "Next Review": "Selects the next turn currently marked as requiring manual review.",
    "Previous Review": "Selects the previous turn currently marked as requiring manual review.",
    "Merge with Next": "Combines the selected turn with the following turn after confirmation.",
    "Split at Final-Text Cursor": (
        "Splits the selected turn at the cursor position in the editable final transcript."
    ),
    "Save Turn": "Saves the current speaker, flags, and final transcript.",
    "Stop Playback": "Stops the currently playing turn audio preview.",
    "Calculate Evaluation": (
        "Calculates Gold Standard evaluation metrics such as WER, CER, speaker accuracy, "
        "and speech-error preservation."
    ),
    "Train and Compare ML Models": (
        "Adds eligible Gold examples, then trains and compares the quality classifiers."
    ),
    "Export Excel": "Exports transcript, evaluation, comparison, and metadata sheets to an Excel workbook.",
    "Export HTML Report": "Exports a self-contained HTML evaluation report with tables and charts.",
    "Apply": "Applies the selected speaker-role mappings to the project.",
    "Cancel": "Closes this dialog without applying changes.",
}


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as HH:MM:SS.t for the live run timer."""
    tenths = max(0, int(seconds * 10))
    whole_seconds, tenth = divmod(tenths, 10)
    minutes, second = divmod(whole_seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02d}:{minute:02d}:{second:02d}.{tenth}"


def _format_byte_size(size: int) -> str:
    """Return a compact binary file-size label for the run log."""
    value = float(max(0, size))
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _transcription_progress(text: str) -> tuple[float, str] | None:
    """Convert structured pipeline messages into global seven-stage progress."""
    stage_match = _STAGE_PROGRESS_RE.search(text)
    if not stage_match:
        return None
    stage = int(stage_match.group(1))
    reported_total = int(stage_match.group(2))
    total = 7 if reported_total in (5, 7) else reported_total
    if total <= 0 or stage <= 0:
        return None

    fraction = 0.0
    item_match = _ITEM_PROGRESS_RE.search(text)
    if item_match:
        item = int(item_match.group(1))
        item_total = int(item_match.group(2))
        if item_total > 0:
            fraction = min(1.0, max(0.0, item / item_total))
    elif any(
        marker in text.casefold()
        for marker in (" complete", " finished", " removed")
    ):
        fraction = 1.0

    percentage = min(100.0, ((stage - 1 + fraction) / total) * 100.0)
    summary = text.split(" - ", 1)[-1].strip()
    if len(summary) > 150:
        summary = summary[:147].rstrip() + "..."
    return percentage, f"Stage {stage} of {total}: {summary}"


def _should_emit_item_progress(text: str, elapsed_since_update: float) -> bool:
    """Throttle intermediate item updates while preserving boundaries."""
    match = _ITEM_PROGRESS_RE.search(text)
    if not match:
        return True
    item, total = map(int, match.groups())
    return (
        item <= 1
        or item >= total
        or elapsed_since_update >= PROGRESS_UPDATE_INTERVAL_SECONDS
    )


def _word_difference_spans(
    source_text: str,
    final_text: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return character spans for words that are not aligned as equal."""
    source_matches = list(_DIFF_WORD_RE.finditer(source_text))
    final_matches = list(_DIFF_WORD_RE.finditer(final_text))
    source_words = [match.group(0).casefold() for match in source_matches]
    final_words = [match.group(0).casefold() for match in final_matches]
    matcher = SequenceMatcher(None, source_words, final_words, autojunk=False)
    source_spans: list[tuple[int, int]] = []
    final_spans: list[tuple[int, int]] = []
    for tag, source_start, source_end, final_start, final_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        source_spans.extend(
            (match.start(), match.end())
            for match in source_matches[source_start:source_end]
        )
        final_spans.extend(
            (match.start(), match.end())
            for match in final_matches[final_start:final_end]
        )
    return source_spans, final_spans


def _training_record_summary(path: str | Path) -> tuple[int, dict[int, int]]:
    """Return the number of saved examples and their label distribution."""
    target = Path(path)
    if not target.exists():
        return 0, {}
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0, {}
    if not isinstance(raw, list):
        return 0, {}
    distribution: dict[int, int] = {}
    valid_records = 0
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            label = int(item["label"])
            features = item["features"]
        except (KeyError, TypeError, ValueError):
            continue
        if (
            item.get("schema_version") != QUALITY_TRAINING_SCHEMA_VERSION
            or item.get("label_target") != QUALITY_LABEL_TARGET
            or not isinstance(item.get("example_id"), str)
            or len(item["example_id"]) != 64
            or not isinstance(features, list)
            or len(features) != len(FEATURE_NAMES)
            or label not in (0, 1, 2)
        ):
            continue
        valid_records += 1
        distribution[label] = distribution.get(label, 0) + 1
    return valid_records, distribution


def _configure_transcribe_row_resizing(frame: ttk.Frame) -> None:
    """Keep action buttons visible while the Transcribe tab is resized.

    The controls row must not absorb a height deficit. The transcription log is
    the flexible area and can become shorter when the application is restored
    from a maximized state.
    """
    frame.rowconfigure(5, weight=0, minsize=44)
    frame.rowconfigure(8, weight=1, minsize=80)


def _initial_window_bounds(screen_width: int, screen_height: int) -> tuple[int, int, int, int]:
    """Return a centered startup window that stays above desktop taskbars.

    Tk reports the full screen size rather than the usable work area on some
    Windows configurations. Reserving generous horizontal and vertical margins
    prevents the status bar at the bottom of the workbench from being clipped.
    """
    safe_width = max(760, screen_width - 80)
    safe_height = max(560, screen_height - 120)
    width = min(1450, safe_width)
    height = min(900, safe_height)
    x = max(0, (screen_width - width) // 2)
    y = max(0, (screen_height - height) // 2 - 10)
    return width, height, x, y


def _wrap_turn_table_text(text: str, width: int) -> str:
    """Wrap a complete transcript for display without truncating its content."""
    normalized = " ".join(text.split())
    if not normalized:
        return ""
    return textwrap.fill(
        normalized,
        width=max(1, int(width)),
        break_long_words=True,
        break_on_hyphens=False,
    )


def _review_speaker_label(turn: Turn) -> str:
    """Prefer an uploaded label, falling back to the inferred speaker."""
    return speaker_label_for_turn(turn)


def _review_tree_rowheight(line_count: int) -> int:
    """Return a safe shared Treeview row height for wrapped transcript text.

    Tk applies one row height to the entire Treeview. A malformed or unusually
    long turn must therefore not request a multi-thousand-pixel row for every
    item, which can exhaust native Tk resources while a project is opened.
    The complete transcript remains stored in the row and in the editor.
    """
    bounded_lines = min(MAX_REVIEW_TREE_LINES, max(1, int(line_count)))
    return max(28, 8 + bounded_lines * 18)


def _evaluate_project_snapshot(project: ProjectData) -> dict[str, object]:
    """Calculate all Tab 4 metrics without reading or updating Tk widgets."""
    metrics: dict[str, object] = dict(evaluate_turns(project.turns))
    source_comparison = per_source_metrics(project.turns)
    if metrics or source_comparison:
        metrics["source_alignment_approximations"] = sum(
            int(item.get("alignment_approximations", 0))
            for item in source_comparison
        )
    metrics["source_comparison"] = source_comparison
    return metrics


def _relative_review_index(
    turns: list[Turn],
    current_index: int | None,
    direction: int,
) -> int | None:
    """Return the adjacent review turn, wrapping at either end."""
    review_indexes = [
        index for index, turn in enumerate(turns) if turn.manual_review
    ]
    if not review_indexes:
        return None
    if current_index is None:
        return review_indexes[0] if direction > 0 else review_indexes[-1]
    if direction > 0:
        return next(
            (index for index in review_indexes if index > current_index),
            review_indexes[0],
        )
    return next(
        (index for index in reversed(review_indexes) if index < current_index),
        review_indexes[-1],
    )


def _review_position(turns: list[Turn], index: int) -> tuple[int | None, int]:
    review_indexes = [
        turn_index
        for turn_index, turn in enumerate(turns)
        if turn.manual_review
    ]
    try:
        position = review_indexes.index(index) + 1
    except ValueError:
        position = None
    return position, len(review_indexes)


class TranscriptionApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_WINDOW_TITLE)
        self.update_idletasks()
        width, height, x, y = _initial_window_bounds(
            self.winfo_screenwidth(),
            self.winfo_screenheight(),
        )
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.minsize(min(1100, width), min(640, height))
        self.project = ProjectData(metadata=ProjectMetadata())
        self._project_dirty = False
        self._syncing_project_ui = False
        self.predictor: object | None = None
        self.current_turn_index: int | None = None
        self.turn_audio_player = TurnAudioPlayer()
        self.playing_turn_index: int | None = None
        self.playback_after_id: str | None = None
        self.transcription_started_at: float | None = None
        self.transcription_timer_after_id: str | None = None
        self.last_transcription_elapsed = 0.0
        self.evaluation_operation_started_at: float | None = None
        self.evaluation_timer_after_id: str | None = None
        self.last_evaluation_elapsed = 0.0
        self.evaluation_operation_name = ""
        self._evaluation_worker_active = False
        self.project_open_started_at: float | None = None
        self._project_open_in_progress = False
        self._loading_editor = False
        self._saving_editor = False
        self._refreshing_turn_table = False
        self._handling_turn_selection = False
        self._turn_table_rewrap_after_id: str | None = None
        self._review_autosave_after_id: str | None = None
        self._main_thread_id = threading.get_ident()
        self._ui_call_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._ui_queue_after_id: str | None = None
        self._closing = False
        self._build_style()
        self._build_menu()
        self._build_ui()
        self._restore_last_selected_files()
        self._install_project_dirty_tracking()
        install_button_tooltips(self, BUTTON_TOOLTIPS)
        self._load_default_model()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._schedule_ui_queue_poll()

    def _selected_files_from_ui(self) -> dict[str, str]:
        return {
            "audio": self.audio_var.get().strip(),
            "zoom": self.zoom_var.get().strip(),
            "chatgpt": self.chatgpt_var.get().strip(),
            "gold": self.gold_var.get().strip(),
        }

    def _restore_last_selected_files(self) -> None:
        remembered = _load_last_selected_files(_last_selected_files_path())
        self.audio_var.set(remembered.get("audio", ""))
        self.zoom_var.set(remembered.get("zoom", ""))
        self.chatgpt_var.set(remembered.get("chatgpt", ""))
        self.gold_var.set(remembered.get("gold", ""))
        self._sync_metadata_from_ui()
        self._update_input_summary()

    def _remember_selected_files(self) -> None:
        try:
            _save_last_selected_files(
                _last_selected_files_path(),
                self._selected_files_from_ui(),
            )
        except OSError:
            # Preference persistence must never prevent the application closing.
            pass

    def _schedule_ui_queue_poll(self) -> None:
        """Schedule the main-thread dispatcher used by background workers."""
        if self._closing or self._ui_queue_after_id is not None:
            return
        self._ui_queue_after_id = self.after(25, self._drain_ui_call_queue)

    def _post_to_ui(self, callback: Callable[[], None]) -> None:
        """Run *callback* on Tk's owning thread without touching Tk from workers."""
        if threading.get_ident() == self._main_thread_id:
            callback()
            return
        self._ui_call_queue.put(callback)

    def _drain_ui_call_queue(self) -> None:
        """Execute queued GUI work on Tk's main thread."""
        self._ui_queue_after_id = None
        try:
            while True:
                callback = self._ui_call_queue.get_nowait()
                try:
                    callback()
                except Exception as exc:
                    self.report_callback_exception(
                        type(exc),
                        exc,
                        exc.__traceback__,
                    )
        except queue.Empty:
            pass

        if not self._closing:
            self._schedule_ui_queue_poll()

    def _speaker_role_choices(self) -> tuple[str, ...]:
        conversation_type = self.conversation_var.get().strip() or "AI"
        roles = list(speaker_roles_for_conversation_type(conversation_type))
        uploaded_labels = sorted(
            {
                label
                for turn in self.project.turns
                if (label := " ".join(turn.speaker_raw.split()).strip())
                and label.casefold() not in {"unknown", "unmapped", "speaker"}
            },
            key=str.casefold,
        )
        detected_names = sorted(
            {
                identity
                for value in (
                    *self.project.speaker_mapping.values(),
                    *(turn.speaker for turn in self.project.turns),
                )
                if (identity := normalize_speaker_identity(value, conversation_type))
                and normalize_role_for_conversation_type(identity, conversation_type) is None
                and identity != "Unknown"
            },
            key=str.casefold,
        )
        choices = dict.fromkeys((*roles, *uploaded_labels, *detected_names, "Unknown"))
        return tuple(choices)

    def _update_speaker_role_choices(self) -> None:
        if not hasattr(self, "speaker_combo"):
            return
        choices = self._speaker_role_choices()
        self.speaker_combo.configure(values=choices)
        selected = self.editor_speaker_var.get().strip()
        preserved = next(
            (choice for choice in choices if choice.casefold() == selected.casefold()),
            None,
        )
        if preserved is not None:
            self.editor_speaker_var.set(preserved)
            return
        current = normalize_speaker_identity(
            selected,
            self.conversation_var.get().strip() or "AI",
        )
        self.editor_speaker_var.set(current or "Unknown")

    def _on_conversation_type_changed(self, _event=None) -> None:
        conversation_type = self.conversation_var.get().strip() or "AI"
        self.project.metadata.conversation_type = conversation_type
        self._update_speaker_role_choices()
        if not self.project.turns:
            return
        automatically_map_speakers(
            self.project,
            status_callback=self._append_log,
        )
        analyze_turns(self.project, self.predictor)
        if self.current_turn_index is not None:
            self.load_turn_into_editor(self.current_turn_index)
        self.refresh_all()
        roles = ", ".join(
            speaker_roles_for_conversation_type(conversation_type)
        )
        self._append_log(
            f"Conversation type changed to {conversation_type}; "
            f"valid fixed speaker roles are now: {roles}; detected learner names are also retained."
        )
        self._set_status("Speaker roles remapped for conversation type")

    def _build_style(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Heading.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Status.TLabel", padding=0)
        style.configure("Treeview", rowheight=28)
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        style.configure("Review.Treeview", rowheight=46)
        style.configure("Review.Treeview.Heading", font=("Segoe UI", 10, "bold"))

    def _build_menu(self) -> None:
        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="New Project", command=self.new_project)
        file_menu.add_command(label="Open Project...", command=self.open_project)
        file_menu.add_command(label="Save Project", command=self.save_project)
        file_menu.add_command(label="Save Project As...", command=lambda: self.save_project(save_as=True))
        file_menu.add_separator()
        file_menu.add_command(label="Export Excel...", command=self.export_excel)
        file_menu.add_command(label="Export Evaluation Report...", command=self.export_report)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menu.add_cascade(label="File", menu=file_menu)

        tools_menu = tk.Menu(menu, tearoff=False)
        tools_menu.add_command(label="Re-align Imported Transcripts", command=self.realign_sources)
        tools_menu.add_command(label="Recalculate Quality Flags", command=self.recalculate_quality)
        tools_menu.add_command(label="Train and Compare Models", command=self.train_models)
        menu.add_cascade(label="Tools", menu=tools_menu)
        self.config(menu=menu)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        outer = ttk.Frame(self, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        ttk.Label(outer, text=APP_TITLE, style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            outer,
            text="Create, compare, review, evaluate, and export turn-level transcripts while preserving learner errors and disfluencies.",
        ).grid(row=1, column=0, sticky="w", pady=(0, 8))

        self.notebook = ttk.Notebook(outer)
        self.notebook.grid(row=2, column=0, sticky="nsew")
        self.project_tab = ttk.Frame(self.notebook, padding=12)
        self.transcribe_tab = ttk.Frame(self.notebook, padding=12)
        self.review_tab = ttk.Frame(self.notebook, padding=8)
        self.evaluation_tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(self.project_tab, text="1. Project Inputs")
        self.notebook.add(self.transcribe_tab, text="2. Transcribe")
        self.notebook.add(self.review_tab, text="3. Review Turns")
        self.notebook.add(self.evaluation_tab, text="4. Evaluate and Export")

        self._build_project_tab()
        self._build_transcribe_tab()
        self._build_review_tab()
        self._build_evaluation_tab()

        self.status_var = tk.StringVar(value="Ready")
        status_frame = ttk.Frame(
            outer,
            relief="sunken",
            borderwidth=1,
            padding=(8, 5),
        )
        status_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        status_frame.columnconfigure(0, weight=1)
        status_frame.columnconfigure(1, weight=0)
        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            style="Status.TLabel",
            anchor="w",
        )
        self.status_label.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            status_frame,
            text=APP_RELEASE_LABEL,
            style="Status.TLabel",
            anchor="e",
        ).grid(row=0, column=1, sticky="e", padx=(16, 0))

    def _build_project_tab(self) -> None:
        frame = self.project_tab
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text="Project metadata", style="Heading.TLabel").grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        self.title_var = tk.StringVar()
        self.learner_var = tk.StringVar()
        self.session_var = tk.StringVar()
        self.conversation_var = tk.StringVar(value="AI")
        metadata_fields = [
            ("Project title", self.title_var),
            ("Learner ID", self.learner_var),
            ("Session number", self.session_var),
        ]
        row = 1
        for label, variable in metadata_fields:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
            ttk.Entry(frame, textvariable=variable).grid(row=row, column=1, columnspan=3, sticky="ew", pady=4)
            row += 1
        ttk.Label(frame, text="Conversation type").grid(row=row, column=0, sticky="w", padx=(0, 10), pady=4)
        self.conversation_combo = ttk.Combobox(
            frame,
            textvariable=self.conversation_var,
            values=("AI", "Human teacher"),
            state="readonly",
            width=25,
        )
        self.conversation_combo.grid(row=row, column=1, sticky="w", pady=4)
        self.conversation_combo.bind(
            "<<ComboboxSelected>>",
            self._on_conversation_type_changed,
        )
        row += 1

        ttk.Separator(frame).grid(row=row, column=0, columnspan=4, sticky="ew", pady=12)
        row += 1
        ttk.Label(frame, text="Files", style="Heading.TLabel").grid(row=row, column=0, columnspan=4, sticky="w", pady=(0, 6))
        row += 1
        self.audio_var = tk.StringVar()
        self.zoom_var = tk.StringVar()
        self.chatgpt_var = tk.StringVar()
        self.gold_var = tk.StringVar()
        file_rows = [
            ("Audio file", self.audio_var, self.browse_audio, AUDIO_FORMAT_NOTE),
            (
                "Zoom transcript",
                self.zoom_var,
                lambda: self.browse_transcript("zoom"),
                TRANSCRIPT_FORMAT_NOTES_BY_SOURCE["zoom"],
            ),
            (
                "ChatGPT transcript",
                self.chatgpt_var,
                lambda: self.browse_transcript("chatgpt"),
                TRANSCRIPT_FORMAT_NOTES_BY_SOURCE["chatgpt"],
            ),
            (
                "Gold Standard transcript",
                self.gold_var,
                lambda: self.browse_transcript("gold"),
                TRANSCRIPT_FORMAT_NOTES_BY_SOURCE["gold"],
            ),
        ]
        for label, variable, command, format_note in file_rows:
            ttk.Label(frame, text=label).grid(
                row=row,
                column=0,
                sticky="w",
                padx=(0, 10),
                pady=4,
            )
            ttk.Entry(frame, textvariable=variable).grid(
                row=row,
                column=1,
                sticky="ew",
                pady=4,
            )
            ttk.Button(frame, text="Browse...", command=command).grid(
                row=row,
                column=2,
                padx=(8, 0),
                pady=4,
            )
            ttk.Label(
                frame,
                text=format_note,
                justify="left",
                wraplength=360,
            ).grid(
                row=row,
                column=3,
                sticky="w",
                padx=(10, 0),
                pady=4,
            )
            row += 1

        buttons = ttk.Frame(frame)
        buttons.grid(row=row, column=0, columnspan=4, sticky="ew", pady=(16, 0))
        ttk.Button(buttons, text="Save Project", command=self.save_project).pack(side="left")
        ttk.Button(buttons, text="Continue to Transcription", command=lambda: self.notebook.select(self.transcribe_tab)).pack(side="right")

        self.input_summary = tk.Text(frame, height=9, wrap="word", state="disabled")
        self.input_summary.grid(row=row + 1, column=0, columnspan=4, sticky="nsew", pady=(14, 0))
        frame.rowconfigure(row + 1, weight=1)

    def _build_transcribe_tab(self) -> None:
        frame = self.transcribe_tab
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=2)
        _configure_transcribe_row_resizing(frame)
        ttk.Label(frame, text="Local additional transcription model", style="Heading.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8)
        )
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.language_var = tk.StringVar(value="auto")
        maximum_threads = available_cpu_threads()
        self.threads_var = tk.IntVar(value=maximum_threads)

        ttk.Label(frame, text="Whisper model").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Combobox(
            frame,
            textvariable=self.model_var,
            state="readonly",
            values=MODEL_CHOICES,
            width=30,
        ).grid(row=1, column=1, sticky="w", pady=4)
        ttk.Label(
            frame,
            text="Recommended: small-q5_1. Larger models improve accuracy but require more RAM and processing time.",
            wraplength=520,
        ).grid(row=1, column=2, sticky="w", padx=(10, 0))

        ttk.Label(frame, text="Language code").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Combobox(
            frame,
            textvariable=self.language_var,
            state="readonly",
            values=LANGUAGE_CHOICES,
            width=28,
        ).grid(row=2, column=1, sticky="w", pady=4)
        ttk.Label(
            frame,
            text=LANGUAGE_CODE_NOTE,
            wraplength=720,
            justify="left",
        ).grid(row=2, column=2, sticky="w", padx=(10, 0))

        ttk.Label(frame, text="CPU threads").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Spinbox(frame, from_=1, to=maximum_threads, textvariable=self.threads_var, width=10).grid(
            row=3, column=1, sticky="w", pady=4
        )

        explanation = (
            "Transcription runs locally with Whisper through pywhispercpp and whisper.cpp. No API key is used, "
            "and the audio is not uploaded. The selected model downloads once on first use and is cached. "
            "Local Whisper does not reliably identify speakers. Imported transcript labels define or identify "
            "the speaking turns. Roles are mapped automatically according to the selected conversation type: "
            "AI and Supervisor for AI conversations, or Teacher for human-teacher conversations. "
            "When the learner's human name appears in an imported transcript label or dialogue, the name is used instead of Student."
        )
        ttk.Label(frame, text=explanation, wraplength=1050, justify="left").grid(
            row=4, column=0, columnspan=3, sticky="nw", pady=(12, 8)
        )
        controls = ttk.Frame(frame)
        controls.grid(
            row=5,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(10, 8),
        )
        controls.columnconfigure(1, weight=1)
        self.transcribe_button = ttk.Button(
            controls,
            text="Run Local Transcription",
            command=self.run_transcription,
        )
        self.transcribe_button.grid(row=0, column=0, sticky="w")
        ttk.Button(
            controls,
            text="Open Review",
            command=lambda: self.notebook.select(self.review_tab),
        ).grid(row=0, column=2, sticky="e")
        run_status = ttk.Frame(frame)
        run_status.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        run_status.columnconfigure(1, weight=1)
        self.transcription_timer_var = tk.StringVar(value="Run time: 00:00:00.0")
        ttk.Label(
            run_status,
            textvariable=self.transcription_timer_var,
            style="Heading.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            run_status,
            text="Times transcript reload, audio preparation, model loading, inference, alignment, and initial analysis.",
        ).grid(row=0, column=1, sticky="e")

        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.grid(row=7, column=0, columnspan=3, sticky="ew")
        self.transcription_log = tk.Text(
            frame,
            height=12,
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
            padx=8,
            pady=8,
        )
        self.transcription_log.grid(row=8, column=0, columnspan=3, sticky="nsew", pady=(10, 0))

    def _build_review_tab(self) -> None:
        frame = self.review_tab
        frame.columnconfigure(0, weight=3)
        frame.columnconfigure(1, weight=2)
        frame.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(frame)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.only_review_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(toolbar, text="Show only turns requiring manual review", variable=self.only_review_var, command=self.refresh_turn_table).pack(side="left")
        ttk.Button(toolbar, text="Previous Review", command=self.select_previous_review).pack(side="left", padx=(8, 4))
        ttk.Button(toolbar, text="Next Review", command=self.select_next_review).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Merge with Next", command=self.merge_with_next).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Split at Final-Text Cursor", command=self.split_current_turn).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Stop Playback", command=self.stop_turn_playback).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Save Turn", command=self.save_editor_to_turn).pack(side="right")

        table_frame = ttk.Frame(frame)
        table_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 7))
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)
        columns = REVIEW_TURN_COLUMNS
        self.turn_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            selectmode="browse",
            style="Review.Treeview",
        )
        headings = {
            "turn": "Turn",
            "time": "Time",
            "speaker": "Speaker",
            "quality": "Quality",
            "listen": "Audio",
            "text": "Final transcript",
        }
        widths = {
            "turn": 55,
            "time": 110,
            "speaker": 95,
            "quality": 150,
            "listen": 75,
            "text": 360,
        }
        for column in columns:
            self.turn_tree.heading(column, text=headings[column])
            self.turn_tree.column(
                column,
                width=widths[column],
                anchor="center" if column == "listen" else "w",
                stretch=column == "text",
            )
        self.turn_tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.turn_tree.yview)
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.turn_tree.configure(yscrollcommand=tree_scroll.set)
        self.turn_tree.bind("<<TreeviewSelect>>", self.on_turn_selected)
        self.turn_tree.bind("<Button-1>", self.on_turn_table_click, add="+")
        self.turn_tree.bind("<Configure>", self._schedule_turn_table_rewrap, add="+")

        editor = ttk.Frame(frame, padding=(7, 0, 0, 0))
        editor.grid(row=1, column=1, sticky="nsew")
        editor.columnconfigure(1, weight=1)
        editor.rowconfigure(3, weight=1)
        self.editor_turn_var = tk.StringVar(value="No turn selected")
        ttk.Label(editor, textvariable=self.editor_turn_var, style="Heading.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(editor, text="Speaker label").grid(row=1, column=0, sticky="w", pady=4)
        self.editor_speaker_var = tk.StringVar()
        self.speaker_combo = ttk.Combobox(
            editor,
            textvariable=self.editor_speaker_var,
            values=self._speaker_role_choices(),
            state="readonly",
        )
        self.speaker_combo.grid(row=1, column=1, sticky="w", pady=4)
        flags = ttk.Frame(editor)
        flags.grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)
        self.hebrew_var = tk.BooleanVar()
        self.hesitation_var = tk.BooleanVar()
        self.self_correction_var = tk.BooleanVar()
        self.unclear_var = tk.BooleanVar()
        self.overlap_var = tk.BooleanVar()
        for text, variable in [
            ("Hebrew switch", self.hebrew_var),
            ("Hesitation/repetition", self.hesitation_var),
            ("Self-correction", self.self_correction_var),
            ("Unclear", self.unclear_var),
            ("Overlap", self.overlap_var),
        ]:
            ttk.Checkbutton(flags, text=text, variable=variable).pack(side="left", padx=(0, 10))
            variable.trace_add("write", self._schedule_review_autosave)
        self.editor_speaker_var.trace_add("write", self._schedule_review_autosave)

        source_stack = ttk.Frame(editor)
        source_stack.grid(
            row=3,
            column=0,
            columnspan=3,
            sticky="nsew",
            pady=6,
        )
        source_stack.columnconfigure(0, weight=1)
        self.source_widgets: dict[str, tk.Text] = {}
        self.active_source_var = tk.StringVar(value="zoom")
        source_items = (
            ("Zoom", "zoom"),
            ("ChatGPT", "chatgpt"),
            ("Additional Model", "model"),
            ("Gold Standard", "gold"),
        )
        for row, (title, key) in enumerate(source_items):
            source_stack.rowconfigure(row, weight=1)
            source_selector = ttk.Radiobutton(
                source_stack,
                text=title,
                value=key,
                variable=self.active_source_var,
                command=self._highlight_source_differences,
            )
            source_box = ttk.LabelFrame(
                source_stack,
                labelwidget=source_selector,
                padding=(4, 2),
            )
            source_box.grid(
                row=row,
                column=0,
                sticky="nsew",
                pady=(0, 4) if row < len(source_items) - 1 else 0,
            )
            source_box.rowconfigure(0, weight=1)
            source_box.columnconfigure(0, weight=1)
            text_widget = tk.Text(
                source_box,
                height=3,
                wrap="word",
                state="disabled",
                padx=4,
                pady=3,
            )
            text_widget.tag_configure("source_difference", background="#ffd9cc")
            text_widget.grid(row=0, column=0, sticky="nsew")
            source_scroll = ttk.Scrollbar(
                source_box,
                orient="vertical",
                command=text_widget.yview,
            )
            source_scroll.grid(row=0, column=1, sticky="ns")
            text_widget.configure(yscrollcommand=source_scroll.set)
            text_widget.bind(
                "<Button-1>",
                lambda _event, source_key=key: self._select_source_box(
                    source_key
                ),
                add="+",
            )
            self.source_widgets[key] = text_widget

        final_heading = ttk.Frame(editor)
        final_heading.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Label(final_heading, text="Final transcript (editable)", style="Heading.TLabel").pack(side="left")
        ttk.Label(
            final_heading,
            text="Highlighted words differ from the selected source box",
        ).pack(side="right")
        self.final_text = tk.Text(editor, height=5, wrap="word", undo=True)
        self.final_text.tag_configure("final_difference", background="#fff0a8")
        self.final_text.grid(row=5, column=0, columnspan=3, sticky="nsew")
        self.final_text.bind("<<Modified>>", self._on_final_text_modified)
        self.final_text.edit_modified(False)

    def _build_evaluation_tab(self) -> None:
        frame = self.evaluation_tab
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(4, weight=3)
        frame.rowconfigure(6, weight=2)

        actions = ttk.Frame(frame)
        actions.grid(row=0, column=0, sticky="ew")
        self.evaluation_action_buttons: list[ttk.Button] = []

        calculate_button = ttk.Button(
            actions,
            text="Calculate Evaluation",
            command=self.calculate_evaluation,
        )
        calculate_button.pack(side="left")
        self.evaluation_action_buttons.append(calculate_button)

        train_button = ttk.Button(
            actions,
            text="Train and Compare ML Models",
            command=self.train_models,
        )
        train_button.pack(side="left")
        self.evaluation_action_buttons.append(train_button)

        excel_button = ttk.Button(
            actions,
            text="Export Excel",
            command=self.export_excel,
        )
        excel_button.pack(side="right")
        self.evaluation_action_buttons.append(excel_button)

        report_button = ttk.Button(
            actions,
            text="Export HTML Report",
            command=self.export_report,
        )
        report_button.pack(side="right", padx=8)
        self.evaluation_action_buttons.append(report_button)

        self.evaluation_progress = ttk.Progressbar(frame, mode="indeterminate")
        self.evaluation_progress.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(8, 0),
        )

        ttk.Label(
            frame,
            text="Evaluation uses the aligned Gold Standard. Speaker accuracy is N/A when the Gold Standard has no usable speaker labels. Speech-error preservation is N/A when no detectable hesitation, repetition, self-correction, unclear marker, or Hebrew word occurs in the Gold Standard. Model training labels the initial transcript shown for review by its Gold-Standard WER and compares class-weighted Logistic Regression, Linear SVM, Random Forest, and a weighted ensemble.",
            wraplength=1100,
        ).grid(row=2, column=0, sticky="w", pady=10)

        ttk.Label(frame, text="Evaluation results", style="Heading.TLabel").grid(
            row=3, column=0, sticky="w", pady=(0, 4)
        )
        results_frame = ttk.Frame(frame)
        results_frame.grid(row=4, column=0, sticky="nsew")
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        self.evaluation_text = tk.Text(results_frame, wrap="word", state="disabled")
        self.evaluation_text.grid(row=0, column=0, sticky="nsew")
        results_scroll = ttk.Scrollbar(
            results_frame,
            orient="vertical",
            command=self.evaluation_text.yview,
        )
        results_scroll.grid(row=0, column=1, sticky="ns")
        self.evaluation_text.configure(yscrollcommand=results_scroll.set)

        log_heading = ttk.Frame(frame)
        log_heading.grid(row=5, column=0, sticky="ew", pady=(10, 4))
        log_heading.columnconfigure(1, weight=1)
        ttk.Label(log_heading, text="Process log", style="Heading.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.evaluation_timer_var = tk.StringVar(
            value="Process time: 00:00:00.0"
        )
        ttk.Label(
            log_heading,
            textvariable=self.evaluation_timer_var,
            style="Heading.TLabel",
        ).grid(row=0, column=1, sticky="e")

        log_frame = ttk.Frame(frame)
        log_frame.grid(row=6, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.evaluation_log = tk.Text(
            log_frame,
            height=10,
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
            padx=8,
            pady=8,
        )
        self.evaluation_log.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(
            log_frame,
            orient="vertical",
            command=self.evaluation_log.yview,
        )
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.evaluation_log.configure(yscrollcommand=log_scroll.set)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _update_window_title(self) -> None:
        project_name = (
            Path(self.project.project_file).name
            if self.project.project_file
            else ""
        )
        title = APP_WINDOW_TITLE
        if project_name:
            title = f"{title} — {project_name}"
        if self._project_dirty:
            title = f"{title} *"
        self.title(title)

    def _set_project_dirty(self, dirty: bool) -> None:
        self._project_dirty = dirty
        self._update_window_title()

    def _mark_project_dirty(self, *_args) -> None:
        if not self._syncing_project_ui and not self._closing:
            self._set_project_dirty(True)

    def _install_project_dirty_tracking(self) -> None:
        for variable in (
            self.title_var,
            self.learner_var,
            self.session_var,
            self.conversation_var,
            self.audio_var,
            self.zoom_var,
            self.chatgpt_var,
            self.gold_var,
            self.model_var,
        ):
            variable.trace_add("write", self._mark_project_dirty)

    def _on_final_text_modified(self, _event=None) -> None:
        if not self.final_text.edit_modified():
            return
        self.final_text.edit_modified(False)
        self._highlight_source_differences()
        self._schedule_review_autosave()

    @staticmethod
    def _apply_text_spans(
        widget: tk.Text,
        tag: str,
        spans: list[tuple[int, int]],
    ) -> None:
        widget.tag_remove(tag, "1.0", "end")
        for start, end in spans:
            widget.tag_add(tag, f"1.0+{start}c", f"1.0+{end}c")

    def _highlight_source_differences(self, _event=None) -> None:
        if not hasattr(self, "final_text"):
            return
        final_text = self.final_text.get("1.0", "end-1c")
        for widget in self.source_widgets.values():
            source_text = widget.get("1.0", "end-1c")
            source_spans, _final_spans = _word_difference_spans(
                source_text,
                final_text,
            )
            self._apply_text_spans(widget, "source_difference", source_spans)

        active_key = self.active_source_var.get()
        if active_key not in self.source_widgets:
            self._apply_text_spans(self.final_text, "final_difference", [])
            return
        active_text = self.source_widgets[active_key].get("1.0", "end-1c")
        _source_spans, final_spans = _word_difference_spans(
            active_text,
            final_text,
        )
        self._apply_text_spans(self.final_text, "final_difference", final_spans)

    def _select_source_box(self, source_key: str) -> None:
        """Use a visible source box as the final-text comparison reference."""
        self.active_source_var.set(source_key)
        self._highlight_source_differences()

    def _schedule_review_autosave(self, *_args) -> None:
        if self._loading_editor or self._saving_editor or self._closing:
            return
        self._mark_project_dirty()
        self._cancel_review_autosave()
        if self.current_turn_index is not None:
            self._review_autosave_after_id = self.after(
                REVIEW_AUTOSAVE_DELAY_MS,
                self._autosave_review_changes,
            )

    def _cancel_review_autosave(self) -> None:
        if self._review_autosave_after_id is None:
            return
        try:
            self.after_cancel(self._review_autosave_after_id)
        except tk.TclError:
            pass
        self._review_autosave_after_id = None

    def _autosave_review_changes(self) -> None:
        self._review_autosave_after_id = None
        if self._closing or not self.project.project_file:
            return
        try:
            self.save_editor_to_turn(silent=True, refresh_table=False)
            self._sync_metadata_from_ui()
            saved = save_project(self.project, self.project.project_file)
        except Exception as exc:
            self._append_log(f"Autosave failed: {exc}")
            self._set_status("Autosave failed; use Save Project to retry")
            return
        saved_at = datetime.now().strftime("%H:%M:%S")
        self._set_project_dirty(False)
        self._set_status(f"Autosaved {saved.name} at {saved_at}")

    def _append_log(self, text: str) -> None:
        """Append one or more timestamped lines to the transcription log."""
        wall_time = datetime.now().strftime("%H:%M:%S")
        elapsed_prefix = ""
        operation_started_at = (
            self.transcription_started_at
            if self.transcription_started_at is not None
            else self.project_open_started_at
        )
        if operation_started_at is not None:
            elapsed = time.monotonic() - operation_started_at
            elapsed_prefix = f" [+{_format_elapsed(elapsed)}]"
        prefix = f"[{wall_time}]{elapsed_prefix} "
        lines = text.rstrip().splitlines() or [""]
        rendered = "\n".join((prefix + line) if line else "" for line in lines) + "\n"
        self.transcription_log.configure(state="normal")
        self.transcription_log.insert("end", rendered)
        self.transcription_log.see("end")
        self.transcription_log.configure(state="disabled")

    def _append_evaluation_log(self, text: str) -> None:
        """Append timestamped process details to the Evaluate and Export log."""
        if not hasattr(self, "evaluation_log"):
            return
        wall_time = datetime.now().strftime("%H:%M:%S")
        elapsed_prefix = ""
        if self.evaluation_operation_started_at is not None:
            elapsed = time.monotonic() - self.evaluation_operation_started_at
            self.last_evaluation_elapsed = elapsed
            elapsed_prefix = f" [+{_format_elapsed(elapsed)}]"
            self._set_evaluation_timer_text(elapsed)
        prefix = f"[{wall_time}]{elapsed_prefix} "
        lines = text.rstrip().splitlines() or [""]
        rendered_lines = [(prefix + line) if line else "" for line in lines]
        self.evaluation_log.configure(state="normal")
        self.evaluation_log.insert("end", "\n".join(rendered_lines) + "\n")
        self.evaluation_log.see("end")
        self.evaluation_log.configure(state="disabled")

    def _set_evaluation_timer_text(
        self, elapsed: float, outcome: str = ""
    ) -> None:
        """Render the current or final Tab 4 process duration."""
        if not hasattr(self, "evaluation_timer_var"):
            return
        operation = self.evaluation_operation_name.strip()
        suffix_parts = [part for part in (operation, outcome.strip()) if part]
        suffix = f" ({' - '.join(suffix_parts)})" if suffix_parts else ""
        self.evaluation_timer_var.set(
            f"Process time: {_format_elapsed(elapsed)}{suffix}"
        )

    def _start_evaluation_timer(self, name: str) -> float:
        """Start the live timer shown above the Tab 4 process log."""
        if self.evaluation_timer_after_id is not None:
            self.after_cancel(self.evaluation_timer_after_id)
        self.evaluation_operation_started_at = time.monotonic()
        self.last_evaluation_elapsed = 0.0
        self.evaluation_operation_name = name
        self._set_evaluation_timer_text(0.0, "running")
        self._update_evaluation_timer()
        return self.evaluation_operation_started_at

    def _update_evaluation_timer(self) -> None:
        """Refresh the Tab 4 timer while an evaluation/export action is active."""
        if self.evaluation_operation_started_at is None:
            self.evaluation_timer_after_id = None
            return
        self.last_evaluation_elapsed = (
            time.monotonic() - self.evaluation_operation_started_at
        )
        self._set_evaluation_timer_text(self.last_evaluation_elapsed, "running")
        self.evaluation_timer_after_id = self.after(
            100, self._update_evaluation_timer
        )

    def _stop_evaluation_timer(
        self, started_at: float, outcome: str
    ) -> float:
        """Stop the Tab 4 timer and preserve its final elapsed duration."""
        elapsed = max(0.0, time.monotonic() - started_at)
        self.last_evaluation_elapsed = elapsed
        if self.evaluation_timer_after_id is not None:
            self.after_cancel(self.evaluation_timer_after_id)
            self.evaluation_timer_after_id = None
        self.evaluation_operation_started_at = None
        self._set_evaluation_timer_text(elapsed, outcome)
        self.evaluation_operation_name = ""
        return elapsed

    def _begin_evaluation_operation(self, name: str) -> float:
        """Start a visibly separated operation in the Tab 4 process log."""
        started_at = self._start_evaluation_timer(name)
        self._append_evaluation_log("")
        self._append_evaluation_log(f"=== {name.upper()} ===")
        self._append_evaluation_log("Button pressed; operation started.")
        return started_at

    def _finish_evaluation_operation(
        self, name: str, started_at: float, outcome: str = "completed"
    ) -> None:
        elapsed = time.monotonic() - started_at
        self._append_evaluation_log(
            f"{name} {outcome} in {_format_elapsed(elapsed)}."
        )
        self._stop_evaluation_timer(started_at, outcome)

    def _fail_evaluation_operation(
        self, name: str, started_at: float, exc: Exception, details: str
    ) -> None:
        elapsed = time.monotonic() - started_at
        self._append_evaluation_log(f"ERROR: {type(exc).__name__}: {exc}")
        self._append_evaluation_log(
            f"{name} failed after {_format_elapsed(elapsed)}."
        )
        self._append_evaluation_log("Traceback follows:")
        self._append_evaluation_log(details.rstrip())
        self._stop_evaluation_timer(started_at, "failed")

    def _log_evaluation_context(self) -> None:
        turns = self.project.turns
        gold_turns = sum(bool(turn.gold_text.strip()) for turn in turns)
        model_turns = sum(bool(turn.model_text.strip()) for turn in turns)
        paired_turns = sum(
            bool(turn.gold_text.strip() and (turn.quality_target_text or turn.final_text or turn.model_text).strip()) for turn in turns
        )
        review_turns = sum(bool(turn.manual_review) for turn in turns)
        self._append_evaluation_log(
            "Project snapshot: "
            f"turns={len(turns)}, Gold-aligned={gold_turns}, "
            f"additional-model={model_turns}, quality-target/Gold pairs={paired_turns}, "
            f"manual-review={review_turns}."
        )

    def _start_transcription_timer(self) -> None:
        if self.transcription_timer_after_id is not None:
            self.after_cancel(self.transcription_timer_after_id)
        self.transcription_started_at = time.monotonic()
        self.last_transcription_elapsed = 0.0
        self.transcription_timer_var.set("Run time: 00:00:00.0")
        self._update_transcription_timer()

    def _update_transcription_timer(self) -> None:
        if self.transcription_started_at is None:
            self.transcription_timer_after_id = None
            return
        self.last_transcription_elapsed = time.monotonic() - self.transcription_started_at
        self.transcription_timer_var.set(
            f"Run time: {_format_elapsed(self.last_transcription_elapsed)}"
        )
        self.transcription_timer_after_id = self.after(100, self._update_transcription_timer)

    def _stop_transcription_timer(self, outcome: str) -> float:
        if self.transcription_started_at is not None:
            self.last_transcription_elapsed = time.monotonic() - self.transcription_started_at
        if self.transcription_timer_after_id is not None:
            self.after_cancel(self.transcription_timer_after_id)
            self.transcription_timer_after_id = None
        self.transcription_started_at = None
        suffix = f" ({outcome})" if outcome else ""
        self.transcription_timer_var.set(
            f"Run time: {_format_elapsed(self.last_transcription_elapsed)}{suffix}"
        )
        return self.last_transcription_elapsed

    def _log_transcription_configuration(
        self, audio: str, model: str, language: str, threads: int
    ) -> None:
        source = Path(audio)
        try:
            size_label = _format_byte_size(source.stat().st_size)
        except OSError:
            size_label = "size unavailable"
        language_label = language or "auto"
        self._append_log(f"Audio: {source}")
        self._append_log(f"Input size: {size_label}; format: {source.suffix.lower() or 'unknown'}")
        self._append_log(
            f"Whisper model: {model}; language: {language_label}; CPU threads: {threads}"
        )
        self._append_log(
            "Imported source segments: "
            f"Zoom={len(self.project.source_transcripts.get('zoom', []))}, "
            f"ChatGPT={len(self.project.source_transcripts.get('chatgpt', []))}, "
            f"Gold={len(self.project.source_transcripts.get('gold', []))}"
        )
        self._append_log("Processing is local; no audio or transcript text is uploaded.")

    def _run_background(self, worker, on_success=None, on_error=None) -> None:
        self.progress.stop()
        self.progress.configure(mode="indeterminate", value=0)
        self.progress.start(12)
        self.transcribe_button.configure(state="disabled")

        def wrapped() -> None:
            try:
                result = worker()
            except Exception as exc:  # GUI boundary: show a clear error rather than crashing.
                details = traceback.format_exc()
                self._post_to_ui(
                    lambda exc=exc, details=details: self._background_failed(
                        exc, details, on_error
                    )
                )
            else:
                self._post_to_ui(
                    lambda result=result: self._background_succeeded(
                        result, on_success, on_error
                    )
                )

        threading.Thread(target=wrapped, daemon=True).start()

    def _set_evaluation_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in getattr(self, "evaluation_action_buttons", []):
            button.configure(state=state)

    def _run_evaluation_background(
        self,
        worker,
        on_success,
        on_error,
    ) -> None:
        """Run a Tab 4 operation without blocking Tk's event loop."""
        if self._evaluation_worker_active:
            messagebox.showinfo(
                APP_TITLE,
                "Another Evaluate and Export operation is already running.",
            )
            return

        self._evaluation_worker_active = True
        self._set_evaluation_controls_enabled(False)
        self.evaluation_progress.stop()
        self.evaluation_progress.configure(mode="indeterminate", value=0)
        self.evaluation_progress.start(12)

        def wrapped() -> None:
            try:
                result = worker()
            except Exception as exc:
                details = traceback.format_exc()
                self._post_to_ui(
                    lambda exc=exc, details=details: self._evaluation_background_failed(
                        exc,
                        details,
                        on_error,
                    )
                )
            else:
                self._post_to_ui(
                    lambda result=result: self._evaluation_background_succeeded(
                        result,
                        on_success,
                        on_error,
                    )
                )

        threading.Thread(
            target=wrapped,
            name="tab4-worker",
            daemon=True,
        ).start()

    def _evaluation_background_failed(
        self,
        exc: Exception,
        details: str,
        on_error,
    ) -> None:
        self.evaluation_progress.stop()
        self._evaluation_worker_active = False
        self._set_evaluation_controls_enabled(True)
        try:
            on_error(exc, details)
        except Exception:
            self._append_evaluation_log(traceback.format_exc())
            self._set_status("Evaluate and Export operation failed")
            messagebox.showerror(APP_TITLE, str(exc))

    def _evaluation_background_succeeded(
        self,
        result,
        on_success,
        on_error,
    ) -> None:
        self.evaluation_progress.stop()
        self._evaluation_worker_active = False
        self._set_evaluation_controls_enabled(True)
        try:
            on_success(result)
        except Exception as exc:
            self._evaluation_background_failed(
                exc,
                traceback.format_exc(),
                on_error,
            )

    def _background_failed(self, exc: Exception, details: str, on_error=None) -> None:
        self.progress.stop()
        self.transcribe_button.configure(state="normal")
        if on_error is not None:
            try:
                on_error(exc, details)
            except Exception:
                self._append_log(traceback.format_exc())
        elif self.transcription_started_at is not None:
            self._append_log(f"ERROR: {exc}")
            elapsed = self._stop_transcription_timer("failed")
            self._append_log(
                f"TRANSCRIPTION RUN FAILED after {_format_elapsed(elapsed)}. "
                "The traceback follows for diagnosis."
            )
            self._append_log(details)
        else:
            self._append_log(details)
        self._set_status("Operation failed")
        messagebox.showerror(APP_TITLE, str(exc))

    def _background_succeeded(self, result, callback, on_error=None) -> None:
        self.progress.stop()
        self.progress.configure(mode="determinate", maximum=100, value=100)
        self.transcribe_button.configure(state="normal")
        if callback:
            try:
                callback(result)
            except Exception as exc:  # Keep post-processing failures visible in the relevant log.
                self._background_failed(exc, traceback.format_exc(), on_error)

    def _sync_metadata_from_ui(self) -> None:
        metadata = self.project.metadata
        metadata.title = self.title_var.get().strip()
        metadata.learner_id = self.learner_var.get().strip()
        metadata.session_number = self.session_var.get().strip()
        metadata.conversation_type = self.conversation_var.get().strip() or "AI"
        metadata.audio_file = self.audio_var.get().strip()
        metadata.zoom_file = self.zoom_var.get().strip()
        metadata.chatgpt_file = self.chatgpt_var.get().strip()
        metadata.gold_file = self.gold_var.get().strip()
        metadata.transcription_model = self.model_var.get().strip()

    def _sync_ui_from_project(self) -> None:
        metadata = self.project.metadata
        self._syncing_project_ui = True
        try:
            self.title_var.set(metadata.title)
            self.learner_var.set(metadata.learner_id)
            self.session_var.set(metadata.session_number)
            self.conversation_var.set(metadata.conversation_type or "AI")
            self._update_speaker_role_choices()
            self.audio_var.set(metadata.audio_file)
            self.zoom_var.set(metadata.zoom_file)
            self.chatgpt_var.set(metadata.chatgpt_file)
            self.gold_var.set(metadata.gold_file)
            selected_model = metadata.transcription_model if metadata.transcription_model in MODEL_CHOICES else DEFAULT_MODEL
            self.model_var.set(selected_model)
        finally:
            self._syncing_project_ui = False
        self.refresh_all()

    def _update_input_summary(self) -> None:
        lines = [
            f"Audio: {self.project.metadata.audio_file or 'not selected'}",
            f"Zoom transcript segments: {len(self.project.source_transcripts.get('zoom', []))}",
            f"ChatGPT transcript segments: {len(self.project.source_transcripts.get('chatgpt', []))}",
            f"Gold Standard segments: {len(self.project.source_transcripts.get('gold', []))}",
            f"Additional model turns: {len(self.project.turns)}",
            f"Turns requiring review: {sum(turn.manual_review for turn in self.project.turns)}",
        ]
        self.input_summary.configure(state="normal")
        self.input_summary.delete("1.0", "end")
        self.input_summary.insert("1.0", "\n".join(lines))
        self.input_summary.configure(state="disabled")

    def browse_audio(self) -> None:
        path = filedialog.askopenfilename(title="Select audio file", filetypes=AUDIO_FILTERS)
        if path:
            self.stop_turn_playback(silent=True)
            self.audio_var.set(path)
            self.project.metadata.audio_file = path
            self._update_input_summary()

    def browse_transcript(self, source_name: str) -> None:
        path = filedialog.askopenfilename(
            title=f"Select {source_name} transcript",
            filetypes=TRANSCRIPT_FILTERS_BY_SOURCE[source_name],
        )
        if not path:
            return
        variable = {"zoom": self.zoom_var, "chatgpt": self.chatgpt_var, "gold": self.gold_var}[source_name]
        variable.set(path)
        try:
            segments = import_source(self.project, source_name, path)
            if self.project.turns:
                align_all_sources(self.project)
                recover_speaker_mapping(
                    self.project,
                    status_callback=self._append_log,
                )
                analyze_turns(self.project, self.predictor)
            self._set_status(f"Imported {len(segments)} {source_name} transcript segments")
            self.refresh_all()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))

    def run_transcription(self) -> None:
        self._sync_metadata_from_ui()
        self.stop_turn_playback(silent=True)
        audio = self.project.metadata.audio_file
        if not audio:
            messagebox.showwarning(APP_TITLE, "Select an audio file first.")
            self.notebook.select(self.project_tab)
            return

        model = self.model_var.get().strip()
        language = _language_code_from_choice(self.language_var.get())
        threads = self.threads_var.get()
        self._start_transcription_timer()
        self._append_log("=" * 78)
        self._append_log("TRANSCRIPTION RUN STARTED")
        self._append_log(
            "Reloading the selected Zoom, ChatGPT, and Gold Standard transcripts from disk."
        )
        self._set_status("Reloading selected transcripts...")
        try:
            transcript_counts = reload_selected_transcripts(
                self.project,
                {
                    "zoom": self.zoom_var.get(),
                    "chatgpt": self.chatgpt_var.get(),
                    "gold": self.gold_var.get(),
                },
            )
        except Exception as exc:
            self._append_log(f"ERROR: Could not reload selected transcripts: {exc}")
            elapsed = self._stop_transcription_timer("failed")
            self._append_log(
                f"TRANSCRIPTION RUN FAILED after {_format_elapsed(elapsed)} before local inference started."
            )
            self._append_log("=" * 78)
            self._set_status("Transcript reload failed")
            messagebox.showerror(
                APP_TITLE,
                f"Could not reload the selected transcripts:\n\n{exc}",
            )
            return

        self._append_log(
            "Transcript reload complete: "
            f"Zoom={transcript_counts['zoom']}, "
            f"ChatGPT={transcript_counts['chatgpt']}, "
            f"Gold={transcript_counts['gold']}."
        )
        self._log_transcription_configuration(audio, model, language, threads)
        self.refresh_all()
        working_project = ProjectData.from_dict(self.project.to_dict())

        last_item_update_at = 0.0

        def status_update(text: str) -> None:
            nonlocal last_item_update_at
            now = time.monotonic()
            if not _should_emit_item_progress(text, now - last_item_update_at):
                return
            if _ITEM_PROGRESS_RE.search(text):
                last_item_update_at = now

            def apply_update(value: str = text) -> None:
                progress = _transcription_progress(value)
                if progress is not None:
                    percentage, summary = progress
                    self.progress.stop()
                    self.progress.configure(
                        mode="determinate",
                        maximum=100,
                        value=percentage,
                    )
                    elapsed = (
                        time.monotonic() - self.transcription_started_at
                        if self.transcription_started_at is not None
                        else 0.0
                    )
                    self._set_status(
                        f"{percentage:.0f}% — {summary} — "
                        f"elapsed {_format_elapsed(elapsed)}"
                    )
                else:
                    self._set_status(value)
                self._append_log(value)

            self._post_to_ui(apply_update)

        def worker():
            segments, details = create_local_transcription(
                audio,
                model_name=model,
                language=language,
                threads=threads,
                status_callback=status_update,
            )
            initialize_turns_from_model(
                working_project,
                segments,
                status_callback=status_update,
            )
            working_project.metadata.transcription_model = model
            return working_project, segments, details

        self._run_background(worker, self._transcription_finished)

    def _transcription_finished(self, result) -> None:
        processed_project, segments, details = result
        self.project = processed_project
        self._mark_project_dirty()
        self._append_log(
            f"Whisper returned {len(segments)} timestamped segments; "
            "alignment and initial analysis completed in the background."
        )
        speaker_labels = sorted({turn.speaker_raw for turn in self.project.turns})
        review_count = sum(turn.manual_review for turn in self.project.turns)
        self._append_log(
            f"Review-turn creation complete: {len(self.project.turns)} turns; "
            f"{review_count} initially flagged for manual review."
        )
        self._append_log(
            "Speaker scaffold labels: " + (", ".join(speaker_labels) if speaker_labels else "none")
        )
        if details:
            duration = details.get("audio_duration_seconds")
            inference = details.get("inference_seconds")
            if isinstance(duration, (int, float)) and isinstance(inference, (int, float)) and duration > 0:
                self._append_log(
                    f"Inference summary: {_format_elapsed(float(inference))} for "
                    f"{_format_elapsed(float(duration))} of audio "
                    f"(real-time factor {float(inference) / float(duration):.3f}x)."
                )
        elapsed = self._stop_transcription_timer("completed")
        self._append_log(f"TRANSCRIPTION RUN COMPLETED in {_format_elapsed(elapsed)}.")
        self._append_log("=" * 78)
        self._set_status("Transcription and initial analysis complete")
        self.refresh_all()
        self.notebook.select(self.review_tab)

    def _schedule_turn_table_rewrap(self, _event=None) -> None:
        """Rewrap transcript cells after the table width changes."""
        if self._turn_table_rewrap_after_id is not None:
            try:
                self.after_cancel(self._turn_table_rewrap_after_id)
            except tk.TclError:
                pass
        self._turn_table_rewrap_after_id = self.after(120, self._rewrap_turn_table)

    def _rewrap_turn_table(self) -> None:
        self._turn_table_rewrap_after_id = None
        self.refresh_turn_table()

    def refresh_all(self) -> None:
        self._update_input_summary()
        self.refresh_turn_table()
        self.refresh_evaluation()

    def refresh_turn_table(self) -> None:
        """Rebuild the turn list without treating programmatic selection as a click."""
        # A learner name can be discovered on one self-introduction turn while
        # other turns still carry the same raw Unknown placeholder. Propagate the
        # project-wide learner identity before rendering every table row.
        propagate_detected_learner_identity(self.project)

        selection_turn_id = None
        if self.current_turn_index is not None and self.current_turn_index < len(self.project.turns):
            selection_turn_id = self.project.turns[self.current_turn_index].turn_id

        self._refreshing_turn_table = True
        try:
            self.turn_tree.delete(*self.turn_tree.get_children())
            text_column_width = int(self.turn_tree.column("text", "width"))
            wrap_width = max(12, int((text_column_width - 12) / 7))
            visible_rows: list[tuple[int, Turn, str]] = []
            maximum_line_count = 1

            for index, turn in enumerate(self.project.turns):
                if self.only_review_var.get() and not turn.manual_review:
                    continue
                display_text = _wrap_turn_table_text(
                    turn.final_text or turn.model_text,
                    wrap_width,
                )
                maximum_line_count = max(
                    maximum_line_count,
                    len(display_text.splitlines()) if display_text else 1,
                )
                visible_rows.append((index, turn, display_text))

            ttk.Style(self).configure(
                "Review.Treeview",
                rowheight=_review_tree_rowheight(maximum_line_count),
            )

            for index, turn, display_text in visible_rows:
                start = self._format_time(turn.start)
                end = self._format_time(turn.end)
                self.turn_tree.insert(
                    "",
                    "end",
                    iid=str(index),
                    values=(
                        turn.turn_id,
                        f"{start} - {end}",
                        _review_speaker_label(turn),
                        turn.quality_label,
                        "■ Stop" if index == self.playing_turn_index else "▶ Play",
                        display_text,
                    ),
                )
            if selection_turn_id is not None:
                for iid in self.turn_tree.get_children():
                    if self.project.turns[int(iid)].turn_id == selection_turn_id:
                        self.turn_tree.selection_set(iid)
                        self.turn_tree.see(iid)
                        break
        finally:
            self._refreshing_turn_table = False

    @staticmethod
    def _format_time(seconds: float | None) -> str:
        if seconds is None:
            return "--:--"
        minutes, seconds_value = divmod(max(0.0, seconds), 60)
        hours, minutes_value = divmod(int(minutes), 60)
        if hours:
            return f"{hours:02d}:{minutes_value:02d}:{seconds_value:05.2f}"
        return f"{minutes_value:02d}:{seconds_value:05.2f}"

    def on_turn_selected(self, _event=None) -> None:
        """Save the previous turn and open the clicked turn exactly once.

        Saving normally refreshes the Treeview. Doing that from inside
        ``<<TreeviewSelect>>`` causes Tk to emit another selection event while the
        first event is still being processed. The guards and deferred refresh
        below prevent that re-entrant callback cycle.
        """
        if self._refreshing_turn_table or self._handling_turn_selection:
            return

        selection = self.turn_tree.selection()
        if not selection:
            return

        try:
            new_index = int(selection[0])
        except (TypeError, ValueError):
            return

        if new_index < 0 or new_index >= len(self.project.turns):
            return
        if self.current_turn_index == new_index:
            return

        self._handling_turn_selection = True
        try:
            if self.current_turn_index is not None:
                self.save_editor_to_turn(silent=True, refresh_table=False)
            self.load_turn_into_editor(new_index)
            # Refresh only after current_turn_index points to the new turn, so the
            # rebuilt table preserves the user's new selection rather than
            # reselecting the previous row.
            self.refresh_turn_table()
        finally:
            self._handling_turn_selection = False

    def on_turn_table_click(self, event: tk.Event[ttk.Treeview]) -> str | None:
        """Play a row's original audio when its Audio cell is clicked."""
        if self.turn_tree.identify_region(event.x, event.y) != "cell":
            return None
        audio_column = f"#{REVIEW_TURN_COLUMNS.index('listen') + 1}"
        if self.turn_tree.identify_column(event.x) != audio_column:
            return None
        item_id = self.turn_tree.identify_row(event.y)
        if not item_id:
            return "break"
        try:
            index = int(item_id)
        except (TypeError, ValueError):
            return "break"
        self.play_turn_audio(index)
        return "break"

    def play_turn_audio(self, index: int) -> None:
        """Extract and play only the timestamp range of one review turn."""
        if index < 0 or index >= len(self.project.turns):
            return
        if self.playing_turn_index == index:
            self.stop_turn_playback()
            return

        audio_path = self.audio_var.get().strip() or self.project.metadata.audio_file
        if not audio_path:
            messagebox.showinfo(APP_TITLE, "Select an audio file before playing a turn.")
            self.notebook.select(self.project_tab)
            return

        turn = self.project.turns[index]
        self._set_status(f"Preparing audio for turn {turn.turn_id}...")
        try:
            duration = self.turn_audio_player.play(audio_path, turn.start, turn.end)
        except TurnPlaybackError as exc:
            self.playing_turn_index = None
            self.refresh_turn_table()
            messagebox.showerror(APP_TITLE, str(exc))
            self._set_status("Turn playback failed")
            return

        if self.playback_after_id is not None:
            try:
                self.after_cancel(self.playback_after_id)
            except tk.TclError:
                pass
        self.playing_turn_index = index
        self.playback_after_id = self.after(
            max(100, int(duration * 1000) + 250),
            self._playback_finished,
        )
        self.refresh_turn_table()
        self._set_status(f"Playing turn {turn.turn_id}")

    def _playback_finished(self) -> None:
        self.playback_after_id = None
        self.playing_turn_index = None
        self.refresh_turn_table()
        self._set_status("Turn playback finished")

    def stop_turn_playback(self, silent: bool = False) -> None:
        """Stop any active turn preview and reset the table audio labels."""
        self.turn_audio_player.stop()
        if self.playback_after_id is not None:
            try:
                self.after_cancel(self.playback_after_id)
            except tk.TclError:
                pass
            self.playback_after_id = None
        was_playing = self.playing_turn_index is not None
        self.playing_turn_index = None
        self.refresh_turn_table()
        if was_playing and not silent:
            self._set_status("Turn playback stopped")

    def load_turn_into_editor(self, index: int) -> None:
        if index < 0 or index >= len(self.project.turns):
            return
        self._loading_editor = True
        self.current_turn_index = index
        turn = self.project.turns[index]
        review_position, review_count = _review_position(self.project.turns, index)
        review_progress = (
            f"Review item {review_position} of {review_count}"
            if review_position is not None
            else f"Not in review queue | {review_count} review item(s) remaining"
        )
        self.editor_turn_var.set(
            f"Turn {turn.turn_id} | {review_progress} | "
            f"{self._format_time(turn.start)} - {self._format_time(turn.end)} | "
            f"Quality score {turn.quality_score:.3f}"
        )
        turn.speaker = resolve_turn_speaker_identity(self.project, turn)
        self._update_speaker_role_choices()
        self.editor_speaker_var.set(turn.speaker)
        self.hebrew_var.set(turn.hebrew_switch)
        self.hesitation_var.set(turn.hesitation_or_repetition)
        self.self_correction_var.set(turn.self_correction)
        self.unclear_var.set(turn.unclear_speech)
        self.overlap_var.set(turn.overlapping_speech)
        for key, value in {"zoom": turn.zoom_text, "chatgpt": turn.chatgpt_text, "model": turn.model_text, "gold": turn.gold_text}.items():
            widget = self.source_widgets[key]
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", value)
            widget.configure(state="disabled")
        self.final_text.delete("1.0", "end")
        self.final_text.insert("1.0", turn.final_text)
        self.final_text.edit_modified(False)
        self._loading_editor = False
        self._highlight_source_differences()

    def save_editor_to_turn(
        self,
        silent: bool = False,
        *,
        refresh_table: bool = True,
    ) -> None:
        if self.current_turn_index is None or self.current_turn_index >= len(self.project.turns) or self._loading_editor:
            return
        turn = self.project.turns[self.current_turn_index]
        self._saving_editor = True
        try:
            selected_label = " ".join(self.editor_speaker_var.get().split()).strip()
            raw_label = " ".join(turn.speaker_raw.split()).strip()
            if (
                selected_label
                and raw_label
                and selected_label.casefold() == raw_label.casefold()
            ):
                turn.speaker = raw_label
            else:
                selected_identity = normalize_speaker_identity(
                    selected_label,
                    self.project.metadata.conversation_type,
                )
                turn.speaker = selected_identity or "Unknown"
            self.editor_speaker_var.set(turn.speaker)
            turn.hebrew_switch = self.hebrew_var.get()
            turn.hesitation_or_repetition = self.hesitation_var.get()
            turn.self_correction = self.self_correction_var.get()
            turn.unclear_speech = self.unclear_var.get()
            turn.overlapping_speech = self.overlap_var.get()
            turn.final_text = self.final_text.get("1.0", "end").strip()
        finally:
            self._saving_editor = False
        if refresh_table:
            self.refresh_turn_table()
        if not silent:
            self._set_status(f"Saved turn {turn.turn_id}")

    def _select_review_relative(self, direction: int) -> None:
        # Persist any text, speaker, and speech-feature edits before moving.
        if self.current_turn_index is not None:
            self.save_editor_to_turn(silent=True, refresh_table=False)

        index = _relative_review_index(
            self.project.turns,
            self.current_turn_index,
            direction,
        )
        if index is None:
            messagebox.showinfo(
                APP_TITLE,
                "No turns are currently marked for manual review.",
            )
            return

        self._handling_turn_selection = True
        try:
            self.current_turn_index = index
            self.load_turn_into_editor(index)
            self.refresh_turn_table()
        finally:
            self._handling_turn_selection = False

    def select_previous_review(self) -> None:
        self._select_review_relative(-1)

    def select_next_review(self) -> None:
        self._select_review_relative(1)

    def merge_with_next(self) -> None:
        if self.current_turn_index is None or self.current_turn_index >= len(self.project.turns) - 1:
            messagebox.showinfo(APP_TITLE, "Select a turn that has a following turn.")
            return
        self.stop_turn_playback(silent=True)
        self.save_editor_to_turn(silent=True)
        first = self.project.turns[self.current_turn_index]
        second = self.project.turns[self.current_turn_index + 1]
        if not messagebox.askyesno(APP_TITLE, f"Merge turn {first.turn_id} with turn {second.turn_id}?"):
            return
        first.end = second.end
        for attribute in ("zoom_text", "chatgpt_text", "model_text", "gold_text", "final_text", "quality_target_text", "notes"):
            combined = " ".join(part for part in (getattr(first, attribute), getattr(second, attribute)) if part.strip())
            setattr(first, attribute, combined)
        first.manual_review = first.manual_review or second.manual_review
        del self.project.turns[self.current_turn_index + 1]
        self._mark_project_dirty()
        self._renumber_turns()
        analyze_turns(self.project, self.predictor)
        self.refresh_all()
        self.load_turn_into_editor(self.current_turn_index)

    def split_current_turn(self) -> None:
        if self.current_turn_index is None:
            messagebox.showinfo(APP_TITLE, "Select a turn first.")
            return
        self.stop_turn_playback(silent=True)
        self.save_editor_to_turn(silent=True)
        cursor = self.final_text.index("insert")
        offset = int(self.final_text.count("1.0", cursor, "chars")[0])
        turn = self.project.turns[self.current_turn_index]
        text = turn.final_text
        if offset <= 0 or offset >= len(text):
            messagebox.showwarning(APP_TITLE, "Place the cursor inside the final transcript where the turn should be split.")
            return
        first_text = text[:offset].strip()
        second_text = text[offset:].strip()
        if not first_text or not second_text:
            messagebox.showwarning(APP_TITLE, "Both parts must contain text.")
            return

        quality_target = turn.quality_target_text or text
        quality_offset = round(
            len(quality_target) * offset / max(1, len(text))
        )
        first_quality_target = quality_target[:quality_offset].strip()
        second_quality_target = quality_target[quality_offset:].strip()
        if not first_quality_target or not second_quality_target:
            first_quality_target = first_text
            second_quality_target = second_text

        midpoint = None
        if turn.start is not None and turn.end is not None:
            ratio = len(first_text) / max(1, len(text))
            midpoint = turn.start + (turn.end - turn.start) * ratio
        new_turn = Turn(
            turn_id=turn.turn_id + 1,
            start=midpoint,
            end=turn.end,
            speaker_raw=turn.speaker_raw,
            speaker=turn.speaker,
            final_text=second_text,
            quality_target_text=second_quality_target,
            model_text="",
            manual_review=True,
            notes="Split manually from the previous turn.",
        )
        turn.end = midpoint
        turn.final_text = first_text
        turn.quality_target_text = first_quality_target
        turn.manual_review = True
        self.project.turns.insert(self.current_turn_index + 1, new_turn)
        self._mark_project_dirty()
        self._renumber_turns()
        analyze_turns(self.project, self.predictor)
        self.refresh_all()
        self.load_turn_into_editor(self.current_turn_index)

    def _renumber_turns(self) -> None:
        for index, turn in enumerate(self.project.turns, start=1):
            turn.turn_id = index

    def realign_sources(self) -> None:
        if not self.project.turns:
            messagebox.showinfo(APP_TITLE, "Run additional transcription first.")
            return
        self.save_editor_to_turn(silent=True)
        align_all_sources(self.project)
        recover_speaker_mapping(
            self.project,
            status_callback=self._append_log,
        )
        analyze_turns(self.project, self.predictor)
        self._mark_project_dirty()
        self.refresh_all()
        self._set_status("Imported transcripts re-aligned")

    def recalculate_quality(self) -> None:
        if not self.project.turns:
            return
        self.save_editor_to_turn(silent=True)
        analyze_turns(self.project, self.predictor)
        self._mark_project_dirty()
        self.refresh_all()
        self._set_status("Quality flags recalculated")

    def _calculate_evaluation(
        self,
        show_missing_gold: bool,
        *,
        log_details: bool = False,
    ) -> bool:
        """Synchronous compatibility helper used only on the Tk thread."""
        if log_details:
            self._append_evaluation_log("Saving any pending Review Turns edits.")
        self.save_editor_to_turn(silent=True)
        metrics = _evaluate_project_snapshot(self.project)
        self.project.metrics = metrics
        self._mark_project_dirty()
        self.refresh_evaluation()
        self._set_status("Evaluation calculated")
        if log_details:
            self._log_evaluation_metrics(metrics)
        has_gold_metrics = "turns_evaluated" in metrics
        if show_missing_gold and not has_gold_metrics:
            messagebox.showinfo(
                APP_TITLE,
                "Import and align a Gold Standard transcript to calculate WER, CER, and related metrics.",
            )
        return has_gold_metrics

    def _log_evaluation_metrics(self, metrics: dict[str, object]) -> None:
        metric_values = {
            key: value
            for key, value in metrics.items()
            if key != "source_comparison"
        }
        if metric_values:
            self._append_evaluation_log(
                f"Calculated {len(metric_values)} aggregate metric(s)."
            )
            for key, value in metric_values.items():
                if value is None:
                    formatted = "N/A"
                elif isinstance(value, float):
                    formatted = f"{value:.4f}"
                else:
                    formatted = str(value)
                self._append_evaluation_log(
                    f"Metric {key.replace('_', ' ')} = {formatted}."
                )
        else:
            self._append_evaluation_log(
                "No Gold Standard-aligned turns were available; "
                "aggregate metrics remain unavailable."
            )

        source_comparison = metrics.get("source_comparison", [])
        if isinstance(source_comparison, list) and source_comparison:
            self._append_evaluation_log(
                f"Calculated source comparisons for {len(source_comparison)} source(s)."
            )
            for item in source_comparison:
                if not isinstance(item, dict):
                    continue
                self._append_evaluation_log(
                    f"Source {item.get('source', '')}: "
                    f"WER={float(item.get('wer', 0.0)):.4f}, "
                    f"CER={float(item.get('cer', 0.0)):.4f}."
                )
        else:
            self._append_evaluation_log(
                "No source had both transcript text and aligned Gold Standard text."
            )

    def calculate_evaluation(self) -> None:
        operation = "Calculate Evaluation"
        started_at = self._begin_evaluation_operation(operation)
        self._log_evaluation_context()
        try:
            self._append_evaluation_log("Saving any pending Review Turns edits.")
            self.save_editor_to_turn(silent=True)
            snapshot = ProjectData.from_dict(self.project.to_dict())
        except Exception as exc:
            self._fail_evaluation_operation(
                operation,
                started_at,
                exc,
                traceback.format_exc(),
            )
            self._set_status("Evaluation failed")
            messagebox.showerror(APP_TITLE, str(exc))
            return

        self._append_evaluation_log(
            "Calculating WER, CER, edit counts, speaker metrics, speech-error "
            "preservation, and source comparisons in a background thread."
        )
        self._set_status("Calculating evaluation...")

        def worker() -> dict[str, object]:
            return _evaluate_project_snapshot(snapshot)

        def on_success(metrics: dict[str, object]) -> None:
            self.project.metrics = metrics
            self.refresh_evaluation()
            self._log_evaluation_metrics(metrics)
            has_gold_metrics = "turns_evaluated" in metrics
            self._set_status("Evaluation calculated")
            outcome = (
                "completed"
                if has_gold_metrics
                else "completed without Gold Standard metrics"
            )
            self._finish_evaluation_operation(operation, started_at, outcome)
            if not has_gold_metrics:
                messagebox.showinfo(
                    APP_TITLE,
                    "Import and align a Gold Standard transcript to calculate "
                    "WER, CER, and related metrics.",
                )

        def on_error(exc: Exception, details: str) -> None:
            self._fail_evaluation_operation(
                operation,
                started_at,
                exc,
                details,
            )
            self._set_status("Evaluation failed")
            messagebox.showerror(APP_TITLE, str(exc))

        self._run_evaluation_background(worker, on_success, on_error)

    def refresh_evaluation(self) -> None:
        lines: list[str] = []
        metrics = {key: value for key, value in self.project.metrics.items() if key != "source_comparison"}
        if metrics:
            lines.append("EVALUATION METRICS")
            for key, value in metrics.items():
                if value is None:
                    formatted = "N/A"
                else:
                    formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
                lines.append(f"{key.replace('_', ' ').title()}: {formatted}")
        else:
            lines.append("No Gold Standard evaluation is available yet.")
        comparison = self.project.metrics.get("source_comparison", [])
        if comparison:
            lines.append("\nTRANSCRIPTION SOURCE COMPARISON")
            for item in comparison:
                lines.append(f"{item['source']}: WER={item['wer']:.4f}, CER={item['cer']:.4f}")
        if self.project.model_comparison:
            lines.append("\nMACHINE-LEARNING MODEL COMPARISON")
            for item in self.project.model_comparison:
                selected = " [ACTIVE]" if item.get("selected") else ""
                lines.append(
                    f"{item['model']}{selected}: "
                    f"accuracy={item['accuracy']:.4f}, "
                    f"balanced accuracy={float(item.get('balanced_accuracy', 0.0)):.4f}, "
                    f"macro F1={item['macro_f1']:.4f}"
                )
        self.evaluation_text.configure(state="normal")
        self.evaluation_text.delete("1.0", "end")
        self.evaluation_text.insert("1.0", "\n".join(lines))
        self.evaluation_text.configure(state="disabled")

    def _project_support_dir(self) -> Path:
        if self.project.project_file:
            return Path(self.project.project_file).resolve().parent / ".transcription_support"
        return Path.cwd() / ".transcription_support"

    def train_models(self) -> None:
        operation = "Train and Compare ML Models"
        started_at = self._begin_evaluation_operation(operation)
        self._log_evaluation_context()
        support = self._project_support_dir()
        training_path = support / "quality_training.json"
        model_path = support / "quality_model.json"
        self._append_evaluation_log(f"Training-set path: {training_path}")
        self._append_evaluation_log(f"Output model path: {model_path}")
        self._append_evaluation_log(
            "Saving pending Review Turns edits and collecting eligible Gold "
            "examples before the training preflight."
        )
        try:
            self.save_editor_to_turn(silent=True)
            snapshot = ProjectData.from_dict(self.project.to_dict())
        except Exception as exc:
            self._fail_evaluation_operation(
                operation,
                started_at,
                exc,
                traceback.format_exc(),
            )
            self._set_status("Preparing model training failed")
            messagebox.showerror(APP_TITLE, str(exc))
            return

        self._append_evaluation_log(
            "Preflight requirement after automatic collection: at least 9 valid "
            "examples across at least 2 quality classes."
        )
        self._append_evaluation_log(
            "Gold-example collection, preflight, training, and held-out comparison "
            "are running in a background thread."
        )
        self._set_status("Adding eligible Gold examples before model training...")

        def worker() -> dict[str, object]:
            added = append_training_examples(snapshot, training_path)
            record_count, distribution = _training_record_summary(training_path)
            training_result = None
            if record_count >= 9 and len(distribution) >= 2:
                training_result = train_quality_model(training_path, model_path)
            return {
                "added": added,
                "record_count": record_count,
                "distribution": distribution,
                "training_result": training_result,
            }

        def on_success(result: dict[str, object]) -> None:
            added = int(result["added"])
            record_count = int(result["record_count"])
            distribution = result["distribution"]
            self._append_evaluation_log(
                f"Automatically added {added} new Gold example(s)."
            )
            self._append_evaluation_log(
                f"Combined training set: records={record_count}; "
                f"label distribution={distribution or 'none'}."
            )
            training_result = result["training_result"]
            if training_result is None:
                self._set_status("Insufficient ML training examples")
                self._finish_evaluation_operation(
                    operation,
                    started_at,
                    "stopped after automatic Gold-example collection",
                )
                messagebox.showinfo(
                    APP_TITLE,
                    "Eligible Gold examples were added automatically, but model "
                    "training still requires at least 9 labeled examples across "
                    "at least 2 quality classes.\n\n"
                    f"Current training set: {record_count} examples; "
                    f"{len(distribution)} classes.",
                )
                return
            self._append_evaluation_log(
                "Models scheduled: class-weighted Logistic Regression, Linear "
                "SVM, Random Forest, and a validation-weighted ensemble."
            )
            self._training_finished(training_result, started_at, model_path)

        def on_error(exc: Exception, details: str) -> None:
            self._fail_evaluation_operation(
                operation, started_at, exc, details
            )
            self._set_status("Model training failed")
            messagebox.showerror(APP_TITLE, str(exc))

        self._run_evaluation_background(worker, on_success, on_error)

    def _training_finished(
        self, result, started_at: float, model_path: Path
    ) -> None:
        operation = "Train and Compare ML Models"
        self._append_evaluation_log(
            "Training finished; applying the selected model and recalculating turn quality flags."
        )
        self.predictor, comparison = result
        self.project.model_comparison = comparison
        if comparison:
            available_rows = int(comparison[0].get("available_rows", 0))
            training_rows = int(comparison[0].get("training_rows", available_rows))
            self._append_evaluation_log(
                f"Model-training rows used: {training_rows} of {available_rows} "
                "available labeled examples."
            )
        for item in comparison:
            selected = " [selected]" if item.get("selected") else ""
            self._append_evaluation_log(
                f"Model {item['model']}{selected}: "
                f"accuracy={item['accuracy']:.4f}, "
                f"balanced accuracy={float(item.get('balanced_accuracy', 0.0)):.4f}, "
                f"macro F1={item['macro_f1']:.4f}, "
                f"selection score={float(item.get('selection_score', 0.0)):.4f}, "
                f"validation predictions={int(item.get('validation_predictions', 0))}."
            )
        analyze_turns(self.project, self.predictor)
        self._mark_project_dirty()
        self.refresh_all()
        self._set_status("Model comparison complete; best model is active")
        active_name = getattr(self.predictor, "name", "Selected model")
        active_metrics = next((item for item in comparison if item.get("model") == active_name), None)
        self._append_evaluation_log(f"Selected active model: {active_name}.")
        try:
            self._append_evaluation_log(
                f"Saved model file: {model_path} "
                f"({_format_byte_size(model_path.stat().st_size)})."
            )
        except OSError:
            self._append_evaluation_log(
                f"Saved model path: {model_path}; file size is unavailable."
            )
        self._append_evaluation_log(
            f"Recalculated quality flags for {len(self.project.turns)} turn(s)."
        )
        self._finish_evaluation_operation(operation, started_at)
        if active_metrics:
            message = (
                f"Training complete. Active model: {active_name} "
                f"(macro F1 {active_metrics['macro_f1']:.3f}, "
                f"balanced accuracy {float(active_metrics.get('balanced_accuracy', 0.0)):.3f}, "
                f"accuracy {active_metrics['accuracy']:.3f})."
            )
        else:
            message = f"Training complete. Active model: {active_name}."
        messagebox.showinfo(APP_TITLE, message)

    def _load_default_model(self) -> None:
        try:
            self.predictor = load_quality_model_if_available(self._project_support_dir() / "quality_model.json")
        except Exception:
            self.predictor = None

    def new_project(self) -> None:
        if self._project_dirty and not messagebox.askyesno(APP_TITLE, "Start a new project? Unsaved changes will be lost."):
            return
        self.stop_turn_playback(silent=True)
        self._cancel_review_autosave()
        self.project = ProjectData(metadata=ProjectMetadata())
        self.current_turn_index = None
        self.predictor = None
        self._sync_ui_from_project()
        self._set_project_dirty(False)
        self._set_status("New project")

    def open_project(self) -> None:
        if self._project_open_in_progress:
            messagebox.showinfo(APP_TITLE, "A project is already being opened.")
            return
        if self._project_dirty and not messagebox.askyesno(
            APP_TITLE,
            "Open another project? Unsaved changes will be lost.",
        ):
            return

        path = filedialog.askopenfilename(
            title="Open project",
            filetypes=[
                ("Transcription projects", "*.ntproject"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self._cancel_review_autosave()
        self.stop_turn_playback(silent=True)
        self.notebook.select(self.transcribe_tab)
        self.project_open_started_at = time.monotonic()
        self._project_open_in_progress = True
        self.progress.stop()
        self.progress.configure(mode="determinate", maximum=100, value=0)
        self.transcribe_button.configure(state="disabled")
        self._append_log("=" * 78)
        self._append_log("PROJECT OPEN STARTED")
        self._append_log(f"Selected project: {Path(path).resolve()}")
        self._update_project_open_progress(2, "Starting project load...")

        def report_progress(percent: int, text: str) -> None:
            self._post_to_ui(
                lambda percent=percent, text=text: self._update_project_open_progress(
                    percent, text
                )
            )

        def worker():
            source = Path(path).expanduser()
            report_progress(8, "Checking project file...")
            size = source.stat().st_size
            report_progress(
                18,
                f"Reading {source.name} ({_format_byte_size(size)})...",
            )
            loaded_project = load_project(source)
            source_counts = {
                name: len(items)
                for name, items in loaded_project.source_transcripts.items()
            }
            report_progress(
                55,
                f"Decoded {len(loaded_project.turns)} saved turns.",
            )

            warnings: list[str] = []
            support_directory = source.resolve().parent / ".transcription_support"
            model_path = support_directory / "quality_model.json"
            predictor = None
            if model_path.is_file():
                report_progress(65, "Loading the saved quality model...")
                try:
                    predictor = load_quality_model_if_available(model_path)
                except Exception as exc:
                    warnings.append(
                        f"Quality model could not be loaded and was ignored: {exc}"
                    )
            else:
                report_progress(65, "No saved quality model was found.")

            report_progress(76, "Checking referenced input files...")
            references = {
                "Audio": loaded_project.metadata.audio_file,
                "Zoom transcript": loaded_project.metadata.zoom_file,
                "ChatGPT transcript": loaded_project.metadata.chatgpt_file,
                "Gold Standard transcript": loaded_project.metadata.gold_file,
            }
            reference_status: list[tuple[str, str, bool]] = []
            for label, raw_path in references.items():
                if not raw_path:
                    continue
                candidate = Path(raw_path).expanduser()
                if not candidate.is_absolute():
                    candidate = source.resolve().parent / candidate
                exists = candidate.is_file()
                reference_status.append((label, str(candidate), exists))
                if not exists:
                    warnings.append(f"Referenced {label.lower()} is missing: {candidate}")

            report_progress(88, "Preparing the saved project state for display...")
            return {
                "project": loaded_project,
                "predictor": predictor,
                "warnings": warnings,
                "reference_status": reference_status,
                "source_counts": source_counts,
                "file_size": size,
            }

        def wrapped() -> None:
            try:
                result = worker()
            except Exception as exc:
                details = traceback.format_exc()
                self._post_to_ui(
                    lambda exc=exc, details=details: self._project_open_failed(
                        exc, details
                    )
                )
            else:
                self._post_to_ui(
                    lambda result=result: self._project_open_finished(result)
                )

        threading.Thread(
            target=wrapped,
            name="project-open-worker",
            daemon=True,
        ).start()

    def _update_project_open_progress(self, percent: int, text: str) -> None:
        """Update project-open progress on Tk's owning thread."""
        if not self._project_open_in_progress or self._closing:
            return
        bounded = max(0, min(100, int(percent)))
        self.progress.configure(value=bounded)
        self._set_status(f"Opening project: {bounded}% - {text}")
        self._append_log(f"[{bounded:3d}%] {text}")

    def _finish_project_open_controls(self) -> None:
        """Restore shared controls after project opening succeeds or fails."""
        self._project_open_in_progress = False
        self.project_open_started_at = None
        self.progress.stop()
        self.progress.configure(mode="indeterminate", value=0)
        self.transcribe_button.configure(state="normal")

    def _project_open_finished(self, result: dict[str, object]) -> None:
        """Commit a fully loaded project only after the worker has succeeded."""
        if self._closing:
            return
        loaded_project = result["project"]
        if not isinstance(loaded_project, ProjectData):
            self._project_open_failed(
                RuntimeError("The project loader returned an invalid result."),
                "The project loader returned an invalid result.",
            )
            return

        previous_project = self.project
        previous_predictor = self.predictor
        previous_turn_index = self.current_turn_index
        try:
            self._update_project_open_progress(92, "Rendering project data...")
            self.project = loaded_project
            self.predictor = result.get("predictor")
            self.current_turn_index = None
            self._sync_ui_from_project()
        except Exception as exc:
            self.project = previous_project
            self.predictor = previous_predictor
            self.current_turn_index = previous_turn_index
            self._project_open_failed(exc, traceback.format_exc())
            return

        source_counts = result.get("source_counts", {})
        self._append_log(
            f"Project JSON loaded: {len(self.project.turns)} turns, "
            f"{_format_byte_size(int(result.get('file_size', 0)))}."
        )
        if isinstance(source_counts, dict):
            rendered_counts = ", ".join(
                f"{name}={count}" for name, count in sorted(source_counts.items())
            )
            self._append_log(
                "Saved source segments: " + (rendered_counts or "none") + "."
            )

        reference_status = result.get("reference_status", [])
        if isinstance(reference_status, list):
            for item in reference_status:
                if not isinstance(item, tuple) or len(item) != 3:
                    continue
                label, reference_path, exists = item
                state = "available" if exists else "missing"
                self._append_log(f"Referenced {label}: {state} - {reference_path}")

        warnings = result.get("warnings", [])
        if isinstance(warnings, list):
            for warning in warnings:
                self._append_log(f"WARNING: {warning}")

        self._append_log(
            "Saved turns and alignments were restored directly. Source transcripts "
            "were not reloaded or re-aligned during opening. Use Tools > "
            "Re-align Imported Transcripts only when you intentionally want to rebuild them."
        )
        self._update_project_open_progress(100, "Project opened successfully.")
        elapsed = (
            time.monotonic() - self.project_open_started_at
            if self.project_open_started_at is not None
            else 0.0
        )
        self._append_log(
            f"PROJECT OPEN COMPLETED in {_format_elapsed(elapsed)}"
        )
        self._append_log("=" * 78)
        self._set_status(f"Opened {Path(self.project.project_file).name}")
        self._set_project_dirty(False)
        self._finish_project_open_controls()

    def _project_open_failed(self, exc: Exception, details: str) -> None:
        """Log project-open failures without replacing the current project."""
        if self._closing:
            return
        elapsed = (
            time.monotonic() - self.project_open_started_at
            if self.project_open_started_at is not None
            else 0.0
        )
        self._append_log(f"ERROR: {exc}")
        self._append_log(
            f"PROJECT OPEN FAILED after {_format_elapsed(elapsed)}. "
            "The current project was left unchanged."
        )
        self._append_log(details)
        self._append_log("=" * 78)
        self._set_status("Project open failed")
        self._finish_project_open_controls()
        messagebox.showerror(
            APP_TITLE,
            f"Could not open the project:\n\n{exc}\n\n"
            "Details were written to the Transcribe log.",
        )

    def save_project(self, save_as: bool = False) -> bool:
        self._cancel_review_autosave()
        self.save_editor_to_turn(silent=True)
        self._sync_metadata_from_ui()
        path = self.project.project_file
        if save_as or not path:
            path = filedialog.asksaveasfilename(
                title="Save project",
                defaultextension=".ntproject",
                filetypes=[("Transcription projects", "*.ntproject")],
            )
        if not path:
            return False
        try:
            saved = save_project(self.project, path)
            self._set_project_dirty(False)
            self._set_status(f"Saved {saved.name}")
            return True
        except Exception as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return False

    def export_excel(self) -> None:
        operation = "Export Excel"
        started_at = self._begin_evaluation_operation(operation)
        self._log_evaluation_context()
        suggested = (
            f"{self.project.metadata.learner_id or 'transcript'}_"
            f"{self.project.metadata.session_number or 'session'}.xlsx"
        )
        self._append_evaluation_log(f"Suggested filename: {suggested}")
        self._append_evaluation_log("Opening the Excel save dialog.")
        path = filedialog.asksaveasfilename(
            title="Export Excel",
            initialfile=suggested,
            defaultextension=".xlsx",
            filetypes=[("Excel workbooks", "*.xlsx")],
        )
        if not path:
            self._append_evaluation_log(
                "The save dialog was cancelled; no workbook was written."
            )
            self._set_status("Excel export cancelled")
            self._finish_evaluation_operation(
                operation,
                started_at,
                "cancelled",
            )
            return

        try:
            self._append_evaluation_log("Saving any pending Review Turns edits.")
            self.save_editor_to_turn(silent=True)
            snapshot = ProjectData.from_dict(self.project.to_dict())
        except Exception as exc:
            self._fail_evaluation_operation(
                operation,
                started_at,
                exc,
                traceback.format_exc(),
            )
            self._set_status("Excel export failed")
            messagebox.showerror(APP_TITLE, str(exc))
            return

        self._append_evaluation_log(f"Selected output path: {path}")
        self._append_evaluation_log(
            "Evaluation and streamed workbook generation are running in a "
            "background thread. Transcript rows are written directly into the "
            "ZIP archive instead of being retained as one large XML string."
        )
        self._set_status("Exporting Excel workbook...")

        def worker() -> dict[str, object]:
            metrics = _evaluate_project_snapshot(snapshot)
            snapshot.metrics = metrics
            result = export_xlsx(snapshot, path)
            try:
                file_size: int | None = result.stat().st_size
            except OSError:
                file_size = None
            return {
                "metrics": metrics,
                "result": result,
                "file_size": file_size,
            }

        def on_success(payload: dict[str, object]) -> None:
            metrics = payload["metrics"]
            if not isinstance(metrics, dict):
                raise TypeError("The evaluation worker returned invalid metrics.")
            result = payload["result"]
            if not isinstance(result, Path):
                raise TypeError("The Excel exporter returned an invalid path.")
            self.project.metrics = metrics
            self.refresh_evaluation()
            self._log_evaluation_metrics(metrics)
            self._append_evaluation_log(f"Workbook created: {result}")
            file_size = payload.get("file_size")
            if isinstance(file_size, int):
                self._append_evaluation_log(
                    f"Workbook size: {_format_byte_size(file_size)}."
                )
            else:
                self._append_evaluation_log("Workbook size is unavailable.")
            self._set_status(f"Exported {result.name}")
            self._finish_evaluation_operation(operation, started_at)
            messagebox.showinfo(
                APP_TITLE,
                f"Excel workbook created:\n{result}",
            )

        def on_error(exc: Exception, details: str) -> None:
            self._fail_evaluation_operation(
                operation,
                started_at,
                exc,
                details,
            )
            self._set_status("Excel export failed")
            messagebox.showerror(APP_TITLE, str(exc))

        self._run_evaluation_background(worker, on_success, on_error)

    def export_report(self) -> None:
        operation = "Export HTML Report"
        started_at = self._begin_evaluation_operation(operation)
        self._log_evaluation_context()
        suggested = (
            f"{self.project.metadata.learner_id or 'transcript'}_"
            f"{self.project.metadata.session_number or 'session'}_report.html"
        )
        self._append_evaluation_log(f"Suggested filename: {suggested}")
        self._append_evaluation_log("Opening the HTML save dialog.")
        path = filedialog.asksaveasfilename(
            title="Export evaluation report",
            initialfile=suggested,
            defaultextension=".html",
            filetypes=[("HTML report", "*.html")],
        )
        if not path:
            self._append_evaluation_log(
                "The save dialog was cancelled; no report was written."
            )
            self._set_status("HTML report export cancelled")
            self._finish_evaluation_operation(
                operation,
                started_at,
                "cancelled",
            )
            return

        try:
            self._append_evaluation_log("Saving any pending Review Turns edits.")
            self.save_editor_to_turn(silent=True)
            snapshot = ProjectData.from_dict(self.project.to_dict())
        except Exception as exc:
            self._fail_evaluation_operation(
                operation,
                started_at,
                exc,
                traceback.format_exc(),
            )
            self._set_status("HTML report export failed")
            messagebox.showerror(APP_TITLE, str(exc))
            return

        self._append_evaluation_log(f"Selected output path: {path}")
        self._append_evaluation_log(
            "Evaluation and report rendering are running in a background thread."
        )
        self._set_status("Exporting HTML report...")

        def worker() -> dict[str, object]:
            metrics = _evaluate_project_snapshot(snapshot)
            snapshot.metrics = metrics
            result = export_html_report(snapshot, path)
            try:
                file_size: int | None = result.stat().st_size
            except OSError:
                file_size = None
            return {
                "metrics": metrics,
                "result": result,
                "file_size": file_size,
            }

        def on_success(payload: dict[str, object]) -> None:
            metrics = payload["metrics"]
            if not isinstance(metrics, dict):
                raise TypeError("The evaluation worker returned invalid metrics.")
            result = payload["result"]
            if not isinstance(result, Path):
                raise TypeError("The report exporter returned an invalid path.")
            self.project.metrics = metrics
            self.refresh_evaluation()
            self._log_evaluation_metrics(metrics)
            self._append_evaluation_log(f"HTML report created: {result}")
            file_size = payload.get("file_size")
            if isinstance(file_size, int):
                self._append_evaluation_log(
                    f"Report size: {_format_byte_size(file_size)}."
                )
            else:
                self._append_evaluation_log("Report size is unavailable.")
            self._set_status(f"Exported {result.name}")
            self._finish_evaluation_operation(operation, started_at)
            if messagebox.askyesno(APP_TITLE, "Report created. Open it now?"):
                opened = webbrowser.open(result.as_uri())
                self._append_evaluation_log(
                    "Open-in-browser requested; "
                    f"browser launch returned {opened}."
                )
            else:
                self._append_evaluation_log(
                    "Report created; open-in-browser was declined."
                )

        def on_error(exc: Exception, details: str) -> None:
            self._fail_evaluation_operation(
                operation,
                started_at,
                exc,
                details,
            )
            self._set_status("HTML report export failed")
            messagebox.showerror(APP_TITLE, str(exc))

        self._run_evaluation_background(worker, on_success, on_error)

    def _on_close(self) -> None:
        if self._project_dirty:
            answer = messagebox.askyesnocancel(APP_TITLE, "Save the project before closing?")
            if answer is None:
                return
            if answer and not self.save_project():
                return

        self._remember_selected_files()
        self._closing = True
        self._cancel_review_autosave()
        if self._ui_queue_after_id is not None:
            self.after_cancel(self._ui_queue_after_id)
            self._ui_queue_after_id = None
        if self._turn_table_rewrap_after_id is not None:
            self.after_cancel(self._turn_table_rewrap_after_id)
            self._turn_table_rewrap_after_id = None
        if self.transcription_timer_after_id is not None:
            self.after_cancel(self.transcription_timer_after_id)
            self.transcription_timer_after_id = None
        if self.evaluation_timer_after_id is not None:
            self.after_cancel(self.evaluation_timer_after_id)
            self.evaluation_timer_after_id = None
        self.stop_turn_playback(silent=True)
        self.turn_audio_player.close()
        self.destroy()


def run() -> None:
    app = TranscriptionApp()
    app.mainloop()
