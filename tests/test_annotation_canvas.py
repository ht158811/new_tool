from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication

from dataset_manager.domain.models.labelme import LabelMeDocument, LabelMeShape
from dataset_manager.ui.widgets.annotation_canvas import AnnotationCanvas


class AnnotationCanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_renders_rectangle_and_polygon_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "preview.png"
            Image.new("RGB", (160, 120), color=(35, 40, 48)).save(image_path)
            document = LabelMeDocument(
                version="5.0",
                flags={},
                shapes=[
                    LabelMeShape(
                        "box", [[10, 10], [80, 10], [80, 70], [10, 70]],
                        shape_type="rectangle",
                    ),
                    LabelMeShape(
                        "region", [[90, 20], [145, 40], [110, 100]], shape_type="polygon"
                    ),
                ],
                image_path="preview.png",
                image_data=None,
                image_height=120,
                image_width=160,
            )
            canvas = AnnotationCanvas()
            canvas.resize(640, 480)
            canvas.set_content(str(image_path), document)
            output = QPixmap(canvas.size())
            canvas.render(output)
            self.assertFalse(output.isNull())
            self.assertEqual(output.size(), canvas.size())
            interior = output.toImage().pixelColor(50, 50)
            self.assertEqual(
                (interior.red(), interior.green(), interior.blue()),
                (35, 40, 48),
            )

            bounds = canvas._shape_bounding_rect(document.shapes[0])
            self.assertEqual((bounds.width(), bounds.height()), (70.0, 60.0))

            spy = QSignalSpy(canvas.shape_label_clicked)
            QTest.mouseClick(
                canvas,
                Qt.MouseButton.LeftButton,
                pos=QPoint(35, 12),
            )
            self.assertEqual(spy.count(), 1)
            self.assertEqual(spy.at(0), [0, "box"])

            canvas.fit_to(QSize(320, 240))
            self.assertLessEqual(canvas.width(), 320)
            self.assertLessEqual(canvas.height(), 240)
            self.assertGreater(canvas.zoom_percent, 0)


if __name__ == "__main__":
    unittest.main()
