from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QWidget

from dataset_manager.domain.models.labelme import LabelMeDocument, LabelMeShape


class AnnotationCanvas(QWidget):
    shape_label_clicked = Signal(int, str)
    zoom_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap = QPixmap()
        self._document: LabelMeDocument | None = None
        self._show_annotations = True
        self._zoom = 1.0
        self.setMinimumSize(64, 64)

    def set_content(self, image_path: str, document: LabelMeDocument | None) -> None:
        self._pixmap = QPixmap(image_path)
        self._document = document
        if not self._pixmap.isNull():
            self._zoom = 1.0
            self._apply_zoom_size()
        self.update()

    def clear(self) -> None:
        self._pixmap = QPixmap()
        self._document = None
        self.resize(640, 480)
        self.update()

    def set_annotations_visible(self, visible: bool) -> None:
        self._show_annotations = visible
        self.update()

    def fit_to(self, viewport_size: QSize) -> None:
        if self._pixmap.isNull():
            return
        available_width = max(1, viewport_size.width() - 36)
        available_height = max(1, viewport_size.height() - 36)
        self.set_zoom(min(
            available_width / self._pixmap.width(),
            available_height / self._pixmap.height(),
        ))

    def set_zoom(self, zoom: float) -> None:
        if self._pixmap.isNull():
            return
        self._zoom = max(0.05, min(8.0, zoom))
        self._apply_zoom_size()
        self.zoom_changed.emit(round(self._zoom * 100))
        self.update()

    def zoom_by(self, factor: float) -> None:
        self.set_zoom(self._zoom * factor)

    @property
    def zoom_percent(self) -> int:
        return round(self._zoom * 100)

    def _apply_zoom_size(self) -> None:
        width = round(self._pixmap.width() * self._zoom) + 32
        height = round(self._pixmap.height() * self._zoom) + 32
        self.resize(max(self.minimumWidth(), width), max(self.minimumHeight(), height))

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API name
        return QSize(640, 480)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API name
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#11151a"))
        if self._pixmap.isNull():
            painter.setPen(QColor("#9aa4b2"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "请选择图片")
            return

        left, top, scale, target_width, target_height = self._target_geometry()
        target = QRectF(left, top, target_width, target_height)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))
        if not self._show_annotations or self._document is None:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for shape in self._document.shapes:
            self._draw_shape(painter, shape, left, top, scale)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API name
        if self._document is None or self._pixmap.isNull() or not self._show_annotations:
            super().mousePressEvent(event)
            return
        if event.button() is not Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        left, top, scale, _, _ = self._target_geometry()
        for index in range(len(self._document.shapes) - 1, -1, -1):
            shape = self._document.shapes[index]
            label_rect = self._label_rect(shape, left, top, scale)
            if label_rect is not None and label_rect.contains(event.position()):
                self.shape_label_clicked.emit(index, shape.label)
                event.accept()
                return
        super().mousePressEvent(event)

    def _target_geometry(self) -> tuple[float, float, float, float, float]:
        scale = self._zoom
        target_width = self._pixmap.width() * scale
        target_height = self._pixmap.height() * scale
        left = (self.width() - target_width) / 2
        top = (self.height() - target_height) / 2
        return left, top, scale, target_width, target_height

    @staticmethod
    def _shape_bounding_rect(shape: LabelMeShape) -> QRectF | None:
        if not shape.points:
            return None
        xs = [point[0] for point in shape.points]
        ys = [point[1] for point in shape.points]
        return QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    def _label_rect(
        self, shape: LabelMeShape, left: float, top: float, scale: float
    ) -> QRectF | None:
        bounds = self._shape_bounding_rect(shape)
        if bounds is None:
            return None
        font = QFont(self.font().family(), 9, QFont.Weight.DemiBold)
        width = max(44, QFontMetrics(font).horizontalAdvance(shape.label) + 12)
        anchor_x = left + bounds.left() * scale
        anchor_y = top + bounds.top() * scale
        return QRectF(anchor_x, max(0.0, anchor_y - 22), width, 20)

    def _draw_shape(
        self, painter: QPainter, shape: LabelMeShape, left: float, top: float, scale: float
    ) -> None:
        if shape.shape_type not in {"rectangle", "polygon"} or not shape.points:
            return
        color = self._label_color(shape.label)
        pen = QPen(color, 2.0)
        painter.setPen(pen)
        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 45))
        points = [QPointF(left + x * scale, top + y * scale) for x, y in shape.points]
        if shape.shape_type == "rectangle" and len(points) >= 2:
            bounds = self._shape_bounding_rect(shape)
            if bounds is None:
                return
            painter.drawRect(QRectF(
                left + bounds.left() * scale,
                top + bounds.top() * scale,
                bounds.width() * scale,
                bounds.height() * scale,
            ))
        elif shape.shape_type == "polygon" and len(points) >= 3:
            painter.drawPolygon(QPolygonF(points))
        else:
            return
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setFont(QFont(painter.font().family(), 9, QFont.Weight.DemiBold))
        label_rect = self._label_rect(shape, left, top, scale)
        if label_rect is None:
            return
        painter.fillRect(label_rect, QColor(color.red(), color.green(), color.blue(), 190))
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(label_rect.adjusted(5, 0, -3, 0), Qt.AlignmentFlag.AlignVCenter, shape.label)

    @staticmethod
    def _label_color(label: str) -> QColor:
        palette = (
            "#38bdf8", "#fb7185", "#a78bfa", "#34d399", "#fbbf24",
            "#f97316", "#22d3ee", "#e879f9",
        )
        index = sum(ord(character) for character in label) % len(palette)
        return QColor(palette[index])

