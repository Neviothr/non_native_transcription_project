"""Tkinter desktop interface for the transcription review workbench."""

from __future__ import annotations

import json
import os
import queue
import textwrap
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .audio_playback import TurnAudioPlayer, TurnPlaybackError
from .evaluation import evaluate_turns, per_source_metrics
from .models import ProjectData, ProjectMetadata, Turn
from .local_whisper import (
    DEFAULT_MODEL,
    MODEL_CHOICES,
    SUPPORTED_AUDIO_SUFFIXES,
    create_local_transcription,
)
from .reporting import export_html_report
from .storage import load_project, save_project
from .workflow import (
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
    speaker_roles_for_conversation_type,
    train_quality_model,
)
from .xlsx_writer import export_xlsx
from .tooltips import install_button_tooltips


APP_TITLE = "Transcription Review Workbench"
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
    "Add Gold Examples": (
        "Adds aligned model and Gold Standard turns to the local quality-model training set."
    ),
    "Train and Compare ML Models": (
        "Trains and compares Logistic Regression, Linear SVM, and Random Forest quality classifiers."
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
        if not isinstance(features, list):
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


def _review_tree_rowheight(line_count: int) -> int:
    """Return a safe shared Treeview row height for wrapped transcript text.

    Tk applies one row height to the entire Treeview. A malformed or unusually
    long turn must therefore not request a multi-thousand-pixel row for every
    item, which can exhaust native Tk resources while a project is opened.
    The complete transcript remains stored in the row and in the editor.
    """
    bounded_lines = min(MAX_REVIEW_TREE_LINES, max(1, int(line_count)))
    return max(28, 8 + bounded_lines * 18)


class TranscriptionApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.update_idletasks()
        width, height, x, y = _initial_window_bounds(
            self.winfo_screenwidth(),
            self.winfo_screenheight(),
        )
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.minsize(min(1100, width), min(640, height))
        self.project = ProjectData(metadata=ProjectMetadata())
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
        self.project_open_started_at: float | None = None
        self._project_open_in_progress = False
        self._loading_editor = False
        self._refreshing_turn_table = False
        self._handling_turn_selection = False
        self._turn_table_rewrap_after_id: str | None = None
        self._main_thread_id = threading.get_ident()
        self._ui_call_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._ui_queue_after_id: str | None = None
        self._closing = False
        self._build_style()
        self._build_menu()
        self._build_ui()
        install_button_tooltips(self, BUTTON_TOOLTIPS)
        self._load_default_model()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._schedule_ui_queue_poll()

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
        return (*roles, *detected_names, "Unknown")

    def _update_speaker_role_choices(self) -> None:
        if not hasattr(self, "speaker_combo"):
            return
        choices = self._speaker_role_choices()
        self.speaker_combo.configure(values=choices)
        current = normalize_speaker_identity(
            self.editor_speaker_var.get(),
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
        tools_menu.add_command(label="Add Gold Examples to Training Set", command=self.add_training_examples)
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
        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            style="Status.TLabel",
            anchor="w",
        )
        self.status_label.grid(row=0, column=0, sticky="ew")

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
        self.threads_var = tk.IntVar(value=min(8, max(1, os.cpu_count() or 1)))

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
        ttk.Spinbox(frame, from_=1, to=max(1, os.cpu_count() or 1), textvariable=self.threads_var, width=10).grid(
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
        frame.columnconfigure(0, weight=2)
        frame.columnconfigure(1, weight=3)
        frame.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(frame)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.only_review_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(toolbar, text="Show only turns requiring manual review", variable=self.only_review_var, command=self.refresh_turn_table).pack(side="left")
        ttk.Button(toolbar, text="Next Review", command=self.select_next_review).pack(side="left", padx=8)
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
        editor.rowconfigure(5, weight=1)
        self.editor_turn_var = tk.StringVar(value="No turn selected")
        ttk.Label(editor, textvariable=self.editor_turn_var, style="Heading.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(editor, text="Speaker identity").grid(row=1, column=0, sticky="w", pady=4)
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

        self.source_notebook = ttk.Notebook(editor)
        self.source_notebook.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=6)
        editor.rowconfigure(3, weight=1)
        self.source_widgets: dict[str, tk.Text] = {}
        for title, key in [("Zoom", "zoom"), ("ChatGPT", "chatgpt"), ("Additional Model", "model"), ("Gold Standard", "gold")]:
            tab = ttk.Frame(self.source_notebook)
            text_widget = tk.Text(tab, height=6, wrap="word", state="disabled")
            text_widget.pack(fill="both", expand=True)
            self.source_notebook.add(tab, text=title)
            self.source_widgets[key] = text_widget

        ttk.Label(editor, text="Final transcript (editable)", style="Heading.TLabel").grid(row=4, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.final_text = tk.Text(editor, height=8, wrap="word", undo=True)
        self.final_text.grid(row=5, column=0, columnspan=3, sticky="nsew")

    def _build_evaluation_tab(self) -> None:
        frame = self.evaluation_tab
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(3, weight=3)
        frame.rowconfigure(5, weight=2)
        actions = ttk.Frame(frame)
        actions.grid(row=0, column=0, sticky="ew")
        ttk.Button(actions, text="Calculate Evaluation", command=self.calculate_evaluation).pack(side="left")
        ttk.Button(actions, text="Add Gold Examples", command=self.add_training_examples).pack(side="left", padx=8)
        ttk.Button(actions, text="Train and Compare ML Models", command=self.train_models).pack(side="left")
        ttk.Button(actions, text="Export Excel", command=self.export_excel).pack(side="right")
        ttk.Button(actions, text="Export HTML Report", command=self.export_report).pack(side="right", padx=8)
        ttk.Label(
            frame,
            text="Evaluation uses the aligned Gold Standard. Speaker accuracy is N/A when the Gold Standard has no usable speaker labels. Speech-error preservation is N/A when no detectable hesitation, repetition, self-correction, unclear marker, or Hebrew word occurs in the Gold Standard. Model training labels each turn by its model-to-gold WER and compares dependency-free Logistic Regression, Linear SVM, and Random Forest classifiers.",
            wraplength=1100,
        ).grid(row=1, column=0, sticky="w", pady=10)

        ttk.Label(frame, text="Evaluation results", style="Heading.TLabel").grid(
            row=2, column=0, sticky="w", pady=(0, 4)
        )
        results_frame = ttk.Frame(frame)
        results_frame.grid(row=3, column=0, sticky="nsew")
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        self.evaluation_text = tk.Text(results_frame, wrap="word", state="disabled")
        self.evaluation_text.grid(row=0, column=0, sticky="nsew")
        results_scroll = ttk.Scrollbar(
            results_frame, orient="vertical", command=self.evaluation_text.yview
        )
        results_scroll.grid(row=0, column=1, sticky="ns")
        self.evaluation_text.configure(yscrollcommand=results_scroll.set)

        log_heading = ttk.Frame(frame)
        log_heading.grid(row=4, column=0, sticky="ew", pady=(10, 4))
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
        log_frame.grid(row=5, column=0, sticky="nsew")
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
            log_frame, orient="vertical", command=self.evaluation_log.yview
        )
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.evaluation_log.configure(yscrollcommand=log_scroll.set)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

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
            bool(turn.gold_text.strip() and turn.model_text.strip()) for turn in turns
        )
        review_turns = sum(bool(turn.manual_review) for turn in turns)
        self._append_evaluation_log(
            "Project snapshot: "
            f"turns={len(turns)}, Gold-aligned={gold_turns}, "
            f"additional-model={model_turns}, paired-for-training={paired_turns}, "
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

        def status_update(text: str) -> None:
            def apply_update(value: str = text) -> None:
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
                        turn.speaker,
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
        self.editor_turn_var.set(f"Turn {turn.turn_id} | {self._format_time(turn.start)} - {self._format_time(turn.end)} | Quality score {turn.quality_score:.3f}")
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
        self._loading_editor = False

    def save_editor_to_turn(
        self,
        silent: bool = False,
        *,
        refresh_table: bool = True,
    ) -> None:
        if self.current_turn_index is None or self.current_turn_index >= len(self.project.turns) or self._loading_editor:
            return
        turn = self.project.turns[self.current_turn_index]
        selected_identity = normalize_speaker_identity(
            self.editor_speaker_var.get(),
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
        if refresh_table:
            self.refresh_turn_table()
        if not silent:
            self._set_status(f"Saved turn {turn.turn_id}")

    def select_next_review(self) -> None:
        # Persist any text, speaker, and speech-feature edits before moving.
        if self.current_turn_index is not None:
            self.save_editor_to_turn(silent=True, refresh_table=False)

        start = (self.current_turn_index + 1) if self.current_turn_index is not None else 0
        candidates = list(range(start, len(self.project.turns))) + list(range(0, start))
        for index in candidates:
            if not self.project.turns[index].manual_review:
                continue

            self._handling_turn_selection = True
            try:
                self.current_turn_index = index
                self.load_turn_into_editor(index)
                self.refresh_turn_table()
            finally:
                self._handling_turn_selection = False
            return
        messagebox.showinfo(APP_TITLE, "No turns are currently marked for manual review.")

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
        for attribute in ("zoom_text", "chatgpt_text", "model_text", "gold_text", "final_text", "notes"):
            combined = " ".join(part for part in (getattr(first, attribute), getattr(second, attribute)) if part.strip())
            setattr(first, attribute, combined)
        first.manual_review = first.manual_review or second.manual_review
        del self.project.turns[self.current_turn_index + 1]
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
            model_text="",
            manual_review=True,
            notes="Split manually from the previous turn.",
        )
        turn.end = midpoint
        turn.final_text = first_text
        turn.manual_review = True
        self.project.turns.insert(self.current_turn_index + 1, new_turn)
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
        self.refresh_all()
        self._set_status("Imported transcripts re-aligned")

    def recalculate_quality(self) -> None:
        if not self.project.turns:
            return
        self.save_editor_to_turn(silent=True)
        analyze_turns(self.project, self.predictor)
        self.refresh_all()
        self._set_status("Quality flags recalculated")

    def _calculate_evaluation(
        self, show_missing_gold: bool, *, log_details: bool = False
    ) -> bool:
        if log_details:
            self._append_evaluation_log("Saving any pending Review Turns edits.")
        self.save_editor_to_turn(silent=True)

        if log_details:
            self._append_evaluation_log(
                "Calculating aggregate Gold Standard metrics: WER, CER, edit counts, "
                "speaker accuracy, speech-error preservation, and manual-review rate."
            )
        self.project.metrics = evaluate_turns(self.project.turns)
        has_gold_metrics = bool(self.project.metrics)

        if log_details:
            self._append_evaluation_log(
                "Calculating per-source WER and CER for Zoom, ChatGPT, "
                "the additional model, and the final transcript."
            )
        source_comparison = per_source_metrics(self.project.turns)
        self.project.metrics["source_comparison"] = source_comparison
        self.refresh_evaluation()
        self._set_status("Evaluation calculated")

        if log_details:
            metric_values = {
                key: value
                for key, value in self.project.metrics.items()
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
                    "No Gold Standard-aligned turns were available; aggregate metrics remain unavailable."
                )

            if source_comparison:
                self._append_evaluation_log(
                    f"Calculated source comparisons for {len(source_comparison)} source(s)."
                )
                for item in source_comparison:
                    self._append_evaluation_log(
                        f"Source {item['source']}: WER={item['wer']:.4f}, "
                        f"CER={item['cer']:.4f}."
                    )
            else:
                self._append_evaluation_log(
                    "No source had both transcript text and aligned Gold Standard text."
                )

        if show_missing_gold and not has_gold_metrics:
            messagebox.showinfo(
                APP_TITLE,
                "Import and align a Gold Standard transcript to calculate WER, CER, and related metrics.",
            )
        return has_gold_metrics

    def calculate_evaluation(self) -> None:
        operation = "Calculate Evaluation"
        started_at = self._begin_evaluation_operation(operation)
        self._log_evaluation_context()
        try:
            has_gold_metrics = self._calculate_evaluation(
                show_missing_gold=True, log_details=True
            )
        except Exception as exc:
            self._fail_evaluation_operation(
                operation, started_at, exc, traceback.format_exc()
            )
            self._set_status("Evaluation failed")
            messagebox.showerror(APP_TITLE, str(exc))
            return
        outcome = "completed" if has_gold_metrics else "completed without Gold Standard metrics"
        self._finish_evaluation_operation(operation, started_at, outcome)

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
                lines.append(f"{item['model']}: accuracy={item['accuracy']:.4f}, macro F1={item['macro_f1']:.4f}")
        self.evaluation_text.configure(state="normal")
        self.evaluation_text.delete("1.0", "end")
        self.evaluation_text.insert("1.0", "\n".join(lines))
        self.evaluation_text.configure(state="disabled")

    def _project_support_dir(self) -> Path:
        if self.project.project_file:
            return Path(self.project.project_file).resolve().parent / ".transcription_support"
        return Path.cwd() / ".transcription_support"

    def add_training_examples(self) -> None:
        operation = "Add Gold Examples"
        started_at = self._begin_evaluation_operation(operation)
        self._log_evaluation_context()
        path = self._project_support_dir() / "quality_training.json"
        paired_turns = sum(
            bool(turn.gold_text.strip() and turn.model_text.strip())
            for turn in self.project.turns
        )
        existing_count, existing_distribution = _training_record_summary(path)
        self._append_evaluation_log(f"Training-set path: {path}")
        self._append_evaluation_log(
            f"Existing valid records: {existing_count}; "
            f"label distribution: {existing_distribution or 'none'}."
        )
        self._append_evaluation_log(
            f"Candidate aligned model/Gold turns in this project: {paired_turns}."
        )
        try:
            self._append_evaluation_log("Saving any pending Review Turns edits.")
            self.save_editor_to_turn(silent=True)
            self._append_evaluation_log(
                "Extracting quality features, assigning WER-based labels, "
                "deduplicating records, and writing the JSON training set."
            )
            count = append_training_examples(self.project, path)
            final_count, final_distribution = _training_record_summary(path)
        except Exception as exc:
            self._fail_evaluation_operation(
                operation, started_at, exc, traceback.format_exc()
            )
            self._set_status("Adding Gold examples failed")
            messagebox.showerror(APP_TITLE, str(exc))
            return

        self._append_evaluation_log(f"New records added: {count}.")
        self._append_evaluation_log(
            f"Training set now contains {final_count} valid record(s); "
            f"label distribution: {final_distribution or 'none'}."
        )
        try:
            self._append_evaluation_log(
                f"Training-set file size: {_format_byte_size(path.stat().st_size)}."
            )
        except OSError:
            self._append_evaluation_log("Training-set file size is unavailable.")

        if count == 0:
            if paired_turns == 0:
                detail = "no aligned turns contain both additional-model and Gold Standard text"
                message = (
                    "No aligned turns contain both an additional-model transcript "
                    "and Gold Standard text."
                )
            else:
                detail = "all eligible examples were already present"
                message = "All eligible Gold Standard examples are already in the training set."
            self._set_status("No new Gold examples added")
            self._finish_evaluation_operation(operation, started_at, f"completed; {detail}")
            messagebox.showinfo(APP_TITLE, message)
            return

        self._set_status(f"Added {count} Gold training examples")
        self._finish_evaluation_operation(operation, started_at)
        messagebox.showinfo(APP_TITLE, f"Added {count} labeled turns to:\n{path}")

    def train_models(self) -> None:
        operation = "Train and Compare ML Models"
        started_at = self._begin_evaluation_operation(operation)
        self._log_evaluation_context()
        support = self._project_support_dir()
        training_path = support / "quality_training.json"
        model_path = support / "quality_model.json"
        self._append_evaluation_log(f"Training-set path: {training_path}")
        self._append_evaluation_log(f"Output model path: {model_path}")

        if not training_path.exists():
            self._append_evaluation_log(
                "Preflight failed: the training-set file does not exist."
            )
            self._set_status("Model training requires Gold examples")
            self._finish_evaluation_operation(
                operation, started_at, "stopped before training"
            )
            messagebox.showinfo(
                APP_TITLE, "Add Gold Standard examples to the training set first."
            )
            return

        record_count, distribution = _training_record_summary(training_path)
        self._append_evaluation_log(
            f"Loaded training summary: records={record_count}, "
            f"label distribution={distribution or 'none'}."
        )
        self._append_evaluation_log(
            "Preflight requirement: at least 9 valid examples across at least 2 quality classes."
        )
        if record_count < 9 or len(distribution) < 2:
            self._append_evaluation_log(
                "Preflight failed: the saved dataset does not meet the minimum size/class requirement."
            )
            self._set_status("Insufficient ML training examples")
            self._finish_evaluation_operation(
                operation, started_at, "stopped before training"
            )
            messagebox.showinfo(
                APP_TITLE,
                "At least 9 labeled examples across at least 2 quality classes are required.",
            )
            return

        self._append_evaluation_log(
            "Models scheduled: Logistic Regression, Linear SVM, and Random Forest."
        )
        self._append_evaluation_log(
            "Training and held-out comparison are running in a background thread."
        )
        self._set_status("Training and comparing ML models...")

        def worker():
            return train_quality_model(training_path, model_path)

        def on_success(result) -> None:
            self._training_finished(result, started_at, model_path)

        def on_error(exc: Exception, details: str) -> None:
            self._fail_evaluation_operation(
                operation, started_at, exc, details
            )

        self._run_background(worker, on_success, on_error)

    def _training_finished(
        self, result, started_at: float, model_path: Path
    ) -> None:
        operation = "Train and Compare ML Models"
        self._append_evaluation_log(
            "Training finished; applying the selected model and recalculating turn quality flags."
        )
        self.predictor, comparison = result
        self.project.model_comparison = comparison
        for item in comparison:
            self._append_evaluation_log(
                f"Model {item['model']}: accuracy={item['accuracy']:.4f}, "
                f"macro F1={item['macro_f1']:.4f}."
            )
        analyze_turns(self.project, self.predictor)
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
            message = f"Training complete. Active model: {active_name} (macro F1 {active_metrics['macro_f1']:.3f}, accuracy {active_metrics['accuracy']:.3f})."
        else:
            message = f"Training complete. Active model: {active_name}."
        messagebox.showinfo(APP_TITLE, message)

    def _load_default_model(self) -> None:
        try:
            self.predictor = load_quality_model_if_available(self._project_support_dir() / "quality_model.json")
        except Exception:
            self.predictor = None

    def new_project(self) -> None:
        if self.project.turns and not messagebox.askyesno(APP_TITLE, "Start a new project? Unsaved changes will be lost."):
            return
        self.stop_turn_playback(silent=True)
        self.project = ProjectData(metadata=ProjectMetadata())
        self.current_turn_index = None
        self.predictor = None
        self._sync_ui_from_project()
        self._set_status("New project")

    def open_project(self) -> None:
        if self._project_open_in_progress:
            messagebox.showinfo(APP_TITLE, "A project is already being opened.")
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
        try:
            self._append_evaluation_log(
                "Refreshing evaluation metrics before building the workbook."
            )
            self._calculate_evaluation(
                show_missing_gold=False, log_details=True
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
                    operation, started_at, "cancelled"
                )
                return

            self._append_evaluation_log(f"Selected output path: {path}")
            self._append_evaluation_log(
                "Writing sheets: Transcript, Evaluation, Source Comparison, "
                "ML Model Comparison, and Metadata."
            )
            result = export_xlsx(self.project, path)
            self._append_evaluation_log(f"Workbook created: {result}")
            try:
                self._append_evaluation_log(
                    f"Workbook size: {_format_byte_size(result.stat().st_size)}."
                )
            except OSError:
                self._append_evaluation_log("Workbook size is unavailable.")
            self._set_status(f"Exported {result.name}")
            self._finish_evaluation_operation(operation, started_at)
            messagebox.showinfo(APP_TITLE, f"Excel workbook created:\n{result}")
        except Exception as exc:
            self._fail_evaluation_operation(
                operation, started_at, exc, traceback.format_exc()
            )
            self._set_status("Excel export failed")
            messagebox.showerror(APP_TITLE, str(exc))

    def export_report(self) -> None:
        operation = "Export HTML Report"
        started_at = self._begin_evaluation_operation(operation)
        self._log_evaluation_context()
        suggested = (
            f"{self.project.metadata.learner_id or 'transcript'}_"
            f"{self.project.metadata.session_number or 'session'}_report.html"
        )
        try:
            self._append_evaluation_log(
                "Refreshing evaluation metrics before rendering the report."
            )
            self._calculate_evaluation(
                show_missing_gold=False, log_details=True
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
                    operation, started_at, "cancelled"
                )
                return

            self._append_evaluation_log(f"Selected output path: {path}")
            self._append_evaluation_log(
                "Rendering metadata, metric tables, source comparison, "
                "model comparison, and self-contained SVG charts."
            )
            result = export_html_report(self.project, path)
            self._append_evaluation_log(f"HTML report created: {result}")
            try:
                self._append_evaluation_log(
                    f"Report size: {_format_byte_size(result.stat().st_size)}."
                )
            except OSError:
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
        except Exception as exc:
            self._fail_evaluation_operation(
                operation, started_at, exc, traceback.format_exc()
            )
            self._set_status("HTML report export failed")
            messagebox.showerror(APP_TITLE, str(exc))

    def _on_close(self) -> None:
        if self.project.turns:
            answer = messagebox.askyesnocancel(APP_TITLE, "Save the project before closing?")
            if answer is None:
                return
            if answer and not self.save_project():
                return

        self._closing = True
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
