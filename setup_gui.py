"""Graphical setup utility. Run by double-clicking SETUP.bat on Windows."""

from __future__ import annotations

import subprocess
import sys
import threading
import venv
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

ROOT = Path(__file__).resolve().parent
EXPECTED = (3, 14, 6)


class SetupWindow(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Transcription Review Workbench Setup")
        self.geometry("760x500")
        self.resizable(False, False)

        outer = ttk.Frame(self, padding=22)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Transcription Review Workbench Setup", font=("Segoe UI", 17, "bold")).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "This setup creates a private Python environment and installs the two direct dependencies used "
                "for fully local transcription: pywhispercpp and imageio-ffmpeg. No command-line knowledge is required."
            ),
            wraplength=700,
        ).pack(anchor="w", pady=(8, 18))

        actual = sys.version_info[:3]
        version_ok = actual == EXPECTED
        ttk.Label(outer, text=f"Python detected: {actual[0]}.{actual[1]}.{actual[2]}", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(outer, text=f"Required project version: {EXPECTED[0]}.{EXPECTED[1]}.{EXPECTED[2]}").pack(anchor="w")
        ttk.Label(
            outer,
            text="Version check passed." if version_ok else "Launch this setup with Python 3.14.6.",
            foreground="#196127" if version_ok else "#9c1c1c",
        ).pack(anchor="w", pady=(4, 15))

        ttk.Label(
            outer,
            text=(
                "The Whisper speech model itself is not bundled with the project. The application downloads the "
                "selected model once when you first transcribe, then reuses the cached copy."
            ),
            wraplength=700,
        ).pack(anchor="w", pady=(0, 12))

        self.progress = ttk.Progressbar(outer, mode="indeterminate")
        self.progress.pack(fill="x", pady=(10, 8))
        self.status_var = tk.StringVar(value="Ready to set up the project.")
        ttk.Label(outer, textvariable=self.status_var, wraplength=700).pack(anchor="w")

        self.log = tk.Text(outer, height=9, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True, pady=(10, 14))

        button_frame = ttk.Frame(outer)
        button_frame.pack(fill="x", side="bottom")
        self.install_button = ttk.Button(
            button_frame,
            text="Set Up Project",
            command=self.start_setup,
            state="normal" if version_ok else "disabled",
        )
        self.install_button.pack(side="left")
        self.launch_button = ttk.Button(
            button_frame,
            text="Launch Application",
            command=self.launch,
            state="normal" if (ROOT / ".venv" / "Scripts" / "pythonw.exe").exists() else "disabled",
        )
        self.launch_button.pack(side="left", padx=8)
        ttk.Button(button_frame, text="Close", command=self.destroy).pack(side="right")

    def start_setup(self) -> None:
        self.install_button.configure(state="disabled")
        self.launch_button.configure(state="disabled")
        self.progress.start(12)
        self.status_var.set("Creating the isolated Python environment...")
        self._append_log("Setup started.")
        threading.Thread(target=self._setup_worker, daemon=True).start()

    def _setup_worker(self) -> None:
        try:
            venv_dir = ROOT / ".venv"
            venv.EnvBuilder(with_pip=True, clear=False, upgrade=False).create(venv_dir)
            python_exe = venv_dir / "Scripts" / "python.exe"
            if not python_exe.exists():
                raise RuntimeError("The virtual environment was created without a Windows Python executable.")

            self._set_worker_status("Updating the package installer...")
            self._run_logged([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])

            self._set_worker_status("Installing local transcription packages...")
            self._run_logged(
                [str(python_exe), "-m", "pip", "install", "--only-binary=:all:", "-r", str(ROOT / "requirements.txt")]
            )
            self._write_launchers()
        except Exception as exc:
            self.after(0, lambda error=exc: self._failed(error))
        else:
            self.after(0, self._completed)

    def _run_logged(self, command: list[str]) -> None:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
        output = completed.stdout.strip()
        if output:
            (ROOT / "setup.log").write_text(output + "\n", encoding="utf-8")
            self.after(0, lambda text=output: self._append_log(text))
        if completed.returncode != 0:
            raise RuntimeError(
                "Package installation failed. Check setup.log. Confirm that the computer is online and that "
                "Python 3.14.6 is the 64-bit Windows build."
            )

    def _set_worker_status(self, text: str) -> None:
        self.after(0, lambda: self.status_var.set(text))
        self.after(0, lambda: self._append_log(text))

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _write_launchers(self) -> None:
        (ROOT / "RUN.bat").write_text(
            '@echo off\r\ncd /d "%~dp0"\r\nstart "" ".venv\\Scripts\\pythonw.exe" "main.py"\r\n',
            encoding="utf-8",
        )
        (ROOT / "RUN_TESTS.bat").write_text(
            '@echo off\r\ncd /d "%~dp0"\r\n".venv\\Scripts\\python.exe" -m unittest discover -s tests -v\r\npause\r\n',
            encoding="utf-8",
        )

    def _failed(self, exc: Exception) -> None:
        self.progress.stop()
        self.install_button.configure(state="normal")
        self.status_var.set("Setup failed.")
        self._append_log(f"ERROR: {exc}")
        messagebox.showerror("Setup", str(exc))

    def _completed(self) -> None:
        self.progress.stop()
        self.status_var.set("Setup complete. Use RUN.bat to open the graphical application.")
        self._append_log("Setup completed successfully.")
        self.launch_button.configure(state="normal")
        messagebox.showinfo(
            "Setup",
            "Setup completed. Local transcription packages are installed. The selected Whisper model will download on first use.",
        )

    def launch(self) -> None:
        pythonw = ROOT / ".venv" / "Scripts" / "pythonw.exe"
        if not pythonw.exists():
            messagebox.showwarning("Setup", "Run setup first.")
            return
        subprocess.Popen([str(pythonw), str(ROOT / "main.py")], cwd=ROOT)
        self.destroy()


if __name__ == "__main__":
    SetupWindow().mainloop()
