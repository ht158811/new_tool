from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class ModelType(str, Enum):
    YOLO_DETECTION = "yolo_detection"
    YOLO_CLASSIFICATION = "yolo_classification"
    YOLO_SEGMENTATION = "yolo_segmentation"


@dataclass(frozen=True, slots=True)
class ModelTestConfig:
    project_id: str
    dataset_id: str
    model_path: str
    model_type: ModelType
    image_width: int
    image_height: int
    confidence: float
    iou_threshold: float
    max_detections: int
    device: str
    output_json: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["model_type"] = self.model_type.value
        return data


@dataclass(frozen=True, slots=True)
class ModelTestProgress:
    processed: int
    total: int
    succeeded: int
    failed: int
    current_image: str


@dataclass(frozen=True, slots=True)
class ModelTestResult:
    output_json: str
    total: int
    succeeded: int
    failed: int
    cancelled: bool = False
