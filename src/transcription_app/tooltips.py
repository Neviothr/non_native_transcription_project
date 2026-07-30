"""Reusable hover tooltips for Tkinter and ttk widgets."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Final


DEFAULT_TOOLTIP_DELAY_MS: Final = 450
DEFAULT_TOOLTIP_WRAP_LENGTH: Final = 360


class HoverTooltip:
    """Display a small explanatory popup while the pointer rests on a widget."""

    def __init__(
        self,
        widget: tk.Misc,
        text: str,
        *,
        delay_ms: int = DEFAULT_TOOLTIP_DELAY_MS,
        wrap_length: int = DEFAULT_TOOLTIP_WRAP_LENGTH,
    ) -> None:
        self.widget = widget
        self.text = text.strip()
        self.delay_ms = max(0, int(delay_ms))
        self.wrap_length = max(120, int(wrap_length))
        self._after_id: str | None = None
        self._window: tk.Toplevel | None = None

        self.widget.bind("<Enter>", self._schedule, add="+")
        self.widget.bind("<Leave>", self._hide, add="+")
        self.widget.bind("<ButtonPress>", self._hide, add="+")
        self.widget.bind("<Destroy>", self._hide, add="+")

    def _schedule(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        self._cancel_schedule()
        if self.text:
            self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel_schedule(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _show(self) -> None:
        self._after_id = None
        if self._window is not None or not self.text:
            return
        try:
            if not self.widget.winfo_exists() or not self.widget.winfo_ismapped():
                return
        except tk.TclError:
            return

        popup = tk.Toplevel(self.widget)
        popup.withdraw()
        popup.overrideredirect(True)
        try:
            popup.attributes("-topmost", True)
        except tk.TclError:
            pass

        label = tk.Label(
            popup,
            text=self.text,
            justify="left",
            relief="solid",
            borderwidth=1,
            background="#fffbe8",
            foreground="#202020",
            font=("Segoe UI", 9),
            padx=7,
            pady=5,
            wraplength=self.wrap_length,
        )
        label.pack()
        popup.update_idletasks()

        x = self.widget.winfo_rootx() + 14
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        screen_width = self.widget.winfo_screenwidth()
        screen_height = self.widget.winfo_screenheight()
        popup_width = popup.winfo_reqwidth()
        popup_height = popup.winfo_reqheight()
        x = max(0, min(x, screen_width - popup_width - 8))
        if y + popup_height > screen_height - 8:
            y = max(0, self.widget.winfo_rooty() - popup_height - 6)

        popup.geometry(f"+{x}+{y}")
        popup.deiconify()
        self._window = popup

    def _hide(self, _event: tk.Event[tk.Misc] | None = None) -> None:
        self._cancel_schedule()
        if self._window is not None:
            try:
                self._window.destroy()
            except tk.TclError:
                pass
            self._window = None


def attach_tooltip(widget: tk.Misc, text: str) -> HoverTooltip:
    """Attach and retain one tooltip instance on a widget."""
    tooltip = HoverTooltip(widget, text)
    setattr(widget, "_hover_tooltip", tooltip)
    return tooltip


def install_button_tooltips(
    parent: tk.Misc,
    descriptions: dict[str, str],
    *,
    fallback_template: str = "Runs the “{text}” action.",
) -> None:
    """Attach tooltips to every existing Tk or ttk button below *parent*."""
    for child in parent.winfo_children():
        if isinstance(child, (ttk.Button, tk.Button)):
            try:
                text = str(child.cget("text")).strip()
            except tk.TclError:
                text = ""
            description = descriptions.get(text)
            if not description:
                description = fallback_template.format(text=text or "selected")
            attach_tooltip(child, description)
        install_button_tooltips(
            child,
            descriptions,
            fallback_template=fallback_template,
        )