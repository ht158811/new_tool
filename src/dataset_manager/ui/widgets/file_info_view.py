from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTextEdit


class FileInfoView(QTextEdit):
    """Read-only file information with vertical and Shift+wheel horizontal scrolling."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            bar = self.horizontalScrollBar()
            delta = event.angleDelta().y() or event.angleDelta().x()
            bar.setValue(bar.value() - delta)
            event.accept()
            return
        super().wheelEvent(event)
