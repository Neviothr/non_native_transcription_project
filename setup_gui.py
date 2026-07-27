"""Graphical setup utility for the Transcription Review Workbench.

Replace the existing setup_gui.py in the project root with this file.

The action buttons are placed in a dedicated bottom bar that never expands or
gets pushed below the visible window. The log panel is the only vertically
resizable part of the interface.
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transcription_app.tooltips import install_button_tooltips

VENV_DIR = PROJECT_DIR / ".venv"
REQUIREMENTS_FILE = PROJECT_DIR / "requirements.txt"
MAIN_FILE = PROJECT_DIR / "main.py"

IS_WINDOWS = os.name == "nt"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

SETUP_BUTTON_TOOLTIPS = {
    "Set Up Project": (
        "Creates the project virtual environment, updates pip, and installs "
        "the packages listed in requirements.txt."
    ),
    "Launch Application": "Starts the Transcription Review Workbench using the project environment.",
    "Close": "Closes the setup window when no setup operation is running.",
}


def get_venv_python() -> Path:
    """Return the Python executable inside the project's virtual environment."""
    if IS_WINDOWS:
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


class SetupGUI:
    """Project setup window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.message_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.setup_running = False

        self.status_var = tk.StringVar(value="Ready.")
        self._configure_window()
        self._build_gui()
        install_button_tooltips(self.root, SETUP_BUTTON_TOOLTIPS)
        self._update_launch_button()
        self._process_messages()

    def _configure_window(self) -> None:
        self.root.title("Transcription Review Workbench Setup")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        width = min(860, max(640, screen_width - 120))
        height = min(640, max(460, screen_height - 140))
        x = max(0, (screen_width - width) // 2)
        y = max(0, (screen_height - height) // 2)

        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(640, 460)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def _build_gui(self) -> None:
        container = ttk.Frame(self.root, padding=(18, 16, 18, 14))
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)

        # Only the log area expands vertically.
        container.rowconfigure(1, weight=1)

        header = ttk.Frame(container)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        header.columnconfigure(0, weight=1)

        ttk.Label(
            header,
            text="Transcription Review Workbench",
            font=("Segoe UI", 17, "bold"),
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(
            header,
            text=(
                "Create the private Python environment and install the packages "
                "required by the project."
            ),
            justify="left",
            wraplength=780,
        ).grid(row=1, column=0, sticky="ew", pady=(5, 0))

        log_frame = ttk.LabelFrame(container, text="Setup progress", padding=8)
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(
            log_frame,
            wrap="word",
            state="disabled",
            height=12,
            font=("Consolas", 9),
            padx=8,
            pady=8,
        )
        log_scrollbar = ttk.Scrollbar(
            log_frame,
            orient="vertical",
            command=self.log_text.yview,
        )
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scrollbar.grid(row=0, column=1, sticky="ns")

        status_frame = ttk.Frame(container)
        status_frame.grid(row=2, column=0, sticky="ew", pady=(10, 8))
        status_frame.columnconfigure(0, weight=1)

        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")

        self.progress = ttk.Progressbar(
            status_frame,
            mode="indeterminate",
        )
        self.progress.grid(row=1, column=0, sticky="ew", pady=(7, 0))

        ttk.Separator(container, orient="horizontal").grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(2, 10),
        )

        # Dedicated bottom action bar. It never receives vertical weight.
        button_bar = ttk.Frame(container)
        button_bar.grid(row=4, column=0, sticky="ew")
        button_bar.columnconfigure(0, weight=1)

        left_buttons = ttk.Frame(button_bar)
        left_buttons.grid(row=0, column=0, sticky="w")

        self.setup_button = ttk.Button(
            left_buttons,
            text="Set Up Project",
            command=self.start_setup,
            width=18,
        )
        self.setup_button.grid(row=0, column=0, padx=(0, 8))

        self.launch_button = ttk.Button(
            left_buttons,
            text="Launch Application",
            command=self.launch_application,
            width=20,
        )
        self.launch_button.grid(row=0, column=1)

        self.close_button = ttk.Button(
            button_bar,
            text="Close",
            command=self.close,
            width=12,
        )
        self.close_button.grid(row=0, column=1, sticky="e", padx=(12, 0))

        self._append_log(f"Project folder: {PROJECT_DIR}")
        self._append_log("Click “Set Up Project” to begin.")

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _post_log(self, message: str) -> None:
        self.message_queue.put(("log", message))

    def _process_messages(self) -> None:
        try:
            while True:
                message_type, value = self.message_queue.get_nowait()

                if message_type == "log":
                    self._append_log(str(value))

                elif message_type == "finished":
                    success = bool(value)
                    self.setup_running = False
                    self.progress.stop()
                    self.setup_button.configure(state="normal")
                    self.close_button.configure(state="normal")
                    self._update_launch_button()

                    if success:
                        self.status_var.set("Setup completed successfully.")
                        messagebox.showinfo(
                            "Setup complete",
                            "The project is ready.",
                            parent=self.root,
                        )
                    else:
                        self.status_var.set("Setup failed. See the progress log.")
                        messagebox.showerror(
                            "Setup failed",
                            "Setup did not complete. See the progress log for details.",
                            parent=self.root,
                        )
        except queue.Empty:
            pass

        self.root.after(100, self._process_messages)

    def _update_launch_button(self) -> None:
        ready = get_venv_python().is_file() and MAIN_FILE.is_file()
        self.launch_button.configure(state="normal" if ready else "disabled")

    def start_setup(self) -> None:
        if self.setup_running:
            return

        self.setup_running = True
        self.setup_button.configure(state="disabled")
        self.launch_button.configure(state="disabled")
        self.close_button.configure(state="disabled")
        self.progress.start(12)
        self.status_var.set("Setting up the project...")

        self._append_log("")
        self._append_log("Starting setup...")

        threading.Thread(target=self._run_setup, daemon=True).start()

    def _run_setup(self) -> None:
        try:
            if sys.version_info[:2] != (3, 14):
                self._post_log(
                    "Warning: the project was designed for Python 3.14.6. "
                    f"Current version: {sys.version.split()[0]}"
                )

            if not VENV_DIR.exists():
                self._post_log("Creating virtual environment...")
                self._run_command(
                    [sys.executable, "-m", "venv", str(VENV_DIR)],
                    "Virtual environment creation",
                )
            else:
                self._post_log("Using the existing virtual environment.")

            python_executable = get_venv_python()
            if not python_executable.is_file():
                raise FileNotFoundError(
                    f"Virtual-environment Python was not found:\n{python_executable}"
                )

            self._post_log("Updating pip...")
            self._run_command(
                [
                    str(python_executable),
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "pip",
                ],
                "pip update",
            )

            if REQUIREMENTS_FILE.is_file():
                requirements_text = REQUIREMENTS_FILE.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                package_lines = [
                    line.strip()
                    for line in requirements_text.splitlines()
                    if line.strip() and not line.lstrip().startswith("#")
                ]

                if package_lines:
                    self._post_log("Installing project packages...")
                    self._run_command(
                        [
                            str(python_executable),
                            "-m",
                            "pip",
                            "install",
                            "-r",
                            str(REQUIREMENTS_FILE),
                        ],
                        "Package installation",
                    )
                else:
                    self._post_log("No third-party packages are listed.")
            else:
                self._post_log(
                    "requirements.txt was not found. Package installation was skipped."
                )

            if not MAIN_FILE.is_file():
                raise FileNotFoundError(
                    f"The application entry point was not found:\n{MAIN_FILE}"
                )

            self._post_log("Setup completed successfully.")
            self.message_queue.put(("finished", True))

        except Exception as exc:
            self._post_log(f"ERROR: {exc}")
            self.message_queue.put(("finished", False))

    def _run_command(self, command: list[str], description: str) -> None:
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

        startup_info = None
        creation_flags = 0

        if IS_WINDOWS:
            startup_info = subprocess.STARTUPINFO()
            startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            creation_flags = CREATE_NO_WINDOW

        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            startupinfo=startup_info,
            creationflags=creation_flags,
        )

        if process.stdout is not None:
            for output_line in process.stdout:
                self._post_log(output_line.rstrip())

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"{description} failed with exit code {return_code}."
            )

    def launch_application(self) -> None:
        python_executable = get_venv_python()

        if not python_executable.is_file():
            messagebox.showerror(
                "Setup required",
                "Run “Set Up Project” first.",
                parent=self.root,
            )
            return

        if not MAIN_FILE.is_file():
            messagebox.showerror(
                "Application not found",
                f"Could not find:\n{MAIN_FILE}",
                parent=self.root,
            )
            return

        try:
            subprocess.Popen(
                [str(python_executable), str(MAIN_FILE)],
                cwd=PROJECT_DIR,
                creationflags=CREATE_NO_WINDOW if IS_WINDOWS else 0,
            )
            self.status_var.set("Application launched.")
            self._append_log("Application launched.")
        except OSError as exc:
            messagebox.showerror(
                "Launch failed",
                str(exc),
                parent=self.root,
            )

    def close(self) -> None:
        if self.setup_running:
            messagebox.showwarning(
                "Setup is running",
                "Wait for setup to finish before closing this window.",
                parent=self.root,
            )
            return

        self.root.destroy()


def main() -> None:
    root = tk.Tk()

    try:
        ttk.Style(root).theme_use("vista")
    except tk.TclError:
        pass

    SetupGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
