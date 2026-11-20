from __future__ import annotations

from typing import Any

from dataset_manager.application.model_testing.models import ModelTestConfig, ModelType


class UltralyticsRunner:
    """Lazy Ultralytics adapter so the rest of the application stays backend-agnostic."""

    def __init__(self) -> None:
        self._model: Any = None

    def load(self, config: ModelTestConfig) -> dict[str, Any]:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "未安装模型测试后端 ultralytics，请重新运行 setup_environment.bat"
            ) from error
        self._model = YOLO(config.model_path)
        expected = {
            ModelType.YOLO_DETECTION: "detect",
            ModelType.YOLO_CLASSIFICATION: "classify",
            ModelType.YOLO_SEGMENTATION: "segment",
        }[config.model_type]
        actual = str(getattr(self._model, "task", "unknown"))
        if actual not in {expected, "unknown", "None"}:
            raise ValueError(f"模型实际类型为 {actual}，与所选类型 {expected} 不一致")
        names = getattr(self._model, "names", {})
        return {"backend": "ultralytics", "task": actual, "classes": self._names(names)}

    def predict(self, image_path: str, config: ModelTestConfig) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("模型尚未加载")
        arguments: dict[str, Any] = {
            "source": image_path,
            "imgsz": (config.image_height, config.image_width),
            "verbose": False,
            "device": config.device.strip() or None,
        }
        if config.model_type is not ModelType.YOLO_CLASSIFICATION:
            arguments.update(
                conf=config.confidence,
                iou=config.iou_threshold,
                max_det=config.max_detections,
            )
        results = self._model.predict(**arguments)
        if not results:
            return {"inference": [], "speed_ms": {}}
        result = results[0]
        payload: dict[str, Any] = {
            "original_shape": list(getattr(result, "orig_shape", ())),
            "speed_ms": {
                str(key): round(float(value), 6)
                for key, value in dict(getattr(result, "speed", {}) or {}).items()
            },
        }
        if config.model_type is ModelType.YOLO_CLASSIFICATION:
            payload["classification"] = self._classification(result)
        else:
            payload["detections"] = self._detections(result)
            if config.model_type is ModelType.YOLO_SEGMENTATION:
                payload["segments"] = self._segments(result)
        return payload

    @staticmethod
    def _names(names: Any) -> dict[str, str]:
        if isinstance(names, dict):
            return {str(key): str(value) for key, value in names.items()}
        if isinstance(names, (list, tuple)):
            return {str(index): str(value) for index, value in enumerate(names)}
        return {}

    def _detections(self, result: Any) -> list[dict[str, Any]]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        names = self._names(getattr(result, "names", {}))
        xyxy = boxes.xyxy.detach().cpu().tolist()
        confidences = boxes.conf.detach().cpu().tolist()
        classes = boxes.cls.detach().cpu().tolist()
        return [
            {
                "class_id": int(class_id),
                "class_name": names.get(str(int(class_id)), str(int(class_id))),
                "confidence": round(float(confidence), 8),
                "box_xyxy": [round(float(value), 4) for value in coordinates],
            }
            for coordinates, confidence, class_id in zip(xyxy, confidences, classes)
        ]

    def _classification(self, result: Any) -> list[dict[str, Any]]:
        probabilities = getattr(result, "probs", None)
        if probabilities is None:
            return []
        names = self._names(getattr(result, "names", {}))
        top_ids = list(getattr(probabilities, "top5", []) or [])
        values = probabilities.data.detach().cpu().tolist()
        return [
            {
                "class_id": int(class_id),
                "class_name": names.get(str(int(class_id)), str(int(class_id))),
                "confidence": round(float(values[int(class_id)]), 8),
            }
            for class_id in top_ids
        ]

    @staticmethod
    def _segments(result: Any) -> list[list[list[float]]]:
        masks = getattr(result, "masks", None)
        polygons = getattr(masks, "xy", None) if masks is not None else None
        if polygons is None:
            return []
        return [
            [[round(float(x), 4), round(float(y), 4)] for x, y in polygon.tolist()]
            for polygon in polygons
        ]
