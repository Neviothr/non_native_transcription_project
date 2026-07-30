from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transcription_app.gui import TranscriptionApp


class _FakeTree:
    def __init__(self, selection: str) -> None:
        self._selection = (selection,)

    def selection(self):
        return self._selection


class _SelectionHarness:
    def __init__(self) -> None:
        self._refreshing_turn_table = False
        self._handling_turn_selection = False
        self.current_turn_index = 0
        self.project = SimpleNamespace(turns=[object(), object()])
        self.turn_tree = _FakeTree("1")
        self.calls: list[object] = []

    def save_editor_to_turn(self, silent=False, *, refresh_table=True) -> None:
        self.calls.append(("save", silent, refresh_table))

    def load_turn_into_editor(self, index: int) -> None:
        self.calls.append(("load", index))
        self.current_turn_index = index

    def refresh_turn_table(self) -> None:
        self.calls.append("refresh")
        # Reproduce the Treeview virtual event emitted by a programmatic
        # selection during a table refresh. The callback must ignore it.
        TranscriptionApp.on_turn_selected(self)


class ReviewSelectionTests(unittest.TestCase):
    def test_clicking_a_different_row_does_not_reenter_selection_callback(self) -> None:
        app = _SelectionHarness()

        TranscriptionApp.on_turn_selected(app)

        self.assertEqual(
            app.calls,
            [
                ("save", True, False),
                ("load", 1),
                "refresh",
            ],
        )
        self.assertEqual(app.current_turn_index, 1)
        self.assertFalse(app._handling_turn_selection)

    def test_selection_event_for_current_row_is_ignored(self) -> None:
        app = _SelectionHarness()
        app.current_turn_index = 1

        TranscriptionApp.on_turn_selected(app)

        self.assertEqual(app.calls, [])


if __name__ == "__main__":
    unittest.main()