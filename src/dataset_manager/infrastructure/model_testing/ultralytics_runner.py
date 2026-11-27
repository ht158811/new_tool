from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

from dataset_manager.application.model_testing.models import ModelTestConfig, ModelType


class UltralyticsRunner:
    """Lazy Ultralytics adapter so the rest of the application stays backend-agnostic."""

    def __init__(self) -> None:
        self._model: Any = None
        self._model_names: dict[str, str] = {}

    def load(self, config: ModelTestConfig) -> dict[str, Any]:
        self._model = None
        self._model_names = {}
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "当前 Python 环境无法导入 ultralytics。"
                f"\n当前解释器：{sys.executable}"
                f"\n原始错误：{type(error).__name__}: {error}"
                "\n请关闭当前软件后，从项目根目录双击 run.bat 启动；"
                "不要使用其他环境中的 dataset-manager 命令。"
            ) from error
        self._validate_backend(config.model_path)
        self._normalize_device(config.device)
        try:
            self._model = YOLO(config.model_path)
        except Exception as error:
            raise RuntimeError(f"无法加载模型文件：{error}") from error
        expected = {
            ModelType.YOLO_DETECTION: "detect",
            ModelType.YOLO_SEGMENTATION: "segment",
        }[config.model_type]
        actual = str(getattr(self._model, "task", "unknown"))
        if actual.casefold() not in {expected, "unknown", "none"}:
            raise ValueError(f"模型实际类型为 {actual}，与所选类型 {expected} 不一致")
        names = getattr(self._model, "names", {})
        self._model_names = self._names(names)
        return {"backend": "ultralytics", "task": actual, "classes": self._model_names}

    def predict(self, image_path: str, config: ModelTestConfig) -> dict[str, Any]:
        if self._model is None:
            raise RuntimeError("模型尚未加载")
        arguments: dict[str, Any] = {
            "source": image_path,
            "imgsz": (config.image_height, config.image_width),
            "verbose": False,
            "conf": config.confidence,
            "iou": config.iou_threshold,
            "max_det": config.max_detections,
            "save": False,
        }
        device = self._normalize_device(config.device)
        if device is not None:
            arguments["device"] = device
        results = self._model.predict(**arguments)
        if not results:
            return {"detections": [], "segments": [], "speed_ms": {}}
        result = results[0]
        payload: dict[str, Any] = {
            "original_shape": list(getattr(result, "orig_shape", ())),
            "speed_ms": {
                str(key): round(float(value), 6)
                for key, value in dict(getattr(result, "speed", {}) or {}).items()
            },
        }
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
        if not names:
            names = self._model_names
        xyxy = self._tolist(getattr(boxes, "xyxy", None))
        confidences = self._tolist(getattr(boxes, "conf", None))
        classes = self._tolist(getattr(boxes, "cls", None))
        return [
            {
                "class_id": int(class_id),
                "class_name": names.get(str(int(class_id)), str(int(class_id))),
                "confidence": round(float(confidence), 8),
                "box_xyxy": [round(float(value), 4) for value in coordinates],
            }
            for coordinates, confidence, class_id in zip(xyxy, confidences, classes)
        ]

    @staticmethod
    def _segments(result: Any) -> list[list[list[float]]]:
        masks = getattr(result, "masks", None)
        polygons = getattr(masks, "xy", None) if masks is not None else None
        if polygons is None:
            return []
        converted: list[list[list[float]]] = []
        for polygon in polygons:
            points = UltralyticsRunner._tolist(polygon)
            converted.append([
                [round(float(point[0]), 4), round(float(point[1]), 4)]
                for point in points
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ])
        return converted

    @staticmethod
    def _tolist(value: Any) -> list[Any]:
        if value is None:
            return []
        for method in ("detach", "cpu"):
            function = getattr(value, method, None)
            if callable(function):
                value = function()
        function = getattr(value, "tolist", None)
        if callable(function):
            value = function()
        return list(value) if isinstance(value, (list, tuple)) else []

    @staticmethod
    def _normalize_device(raw_device: str) -> str | None:
        device = raw_device.strip().casefold()
        if device in {"", "auto", "自动"}:
            return None
        if device == "cpu":
            return "cpu"
        if device == "cuda":
            device = "0"
        elif device.startswith("cuda:"):
            device = device.split(":", 1)[1]
        if all(part.strip().isdigit() for part in device.split(",")):
            try:
                import torch
            except ImportError as error:
                raise RuntimeError("未安装 PyTorch，无法使用 CUDA 设备") from error
            if not torch.cuda.is_available():
                raise ValueError("当前 PyTorch 环境不支持 CUDA，请留空自动选择或输入 cpu")
            indexes = [int(part.strip()) for part in device.split(",")]
            count = torch.cuda.device_count()
            invalid = [index for index in indexes if index < 0 or index >= count]
            if invalid:
                raise ValueError(
                    f"CUDA 设备编号不存在：{invalid}；当前可用 GPU 数量为 {count}"
                )
            return ",".join(str(index) for index in indexes)
        return device

    @staticmethod
    def _validate_backend(model_path: str) -> None:
        suffix = Path(model_path).suffix.casefold()
        requirements = {
            ".onnx": (
                "onnxruntime",
                "ONNX 模型需要安装 onnxruntime 或 onnxruntime-gpu",
            ),
            ".engine": (
                "tensorrt",
                "TensorRT engine 需要 NVIDIA GPU、匹配版本的 CUDA 和 tensorrt",
            ),
        }
        requirement = requirements.get(suffix)
        if requirement is not None and importlib.util.find_spec(requirement[0]) is None:
            raise RuntimeError(requirement[1])
