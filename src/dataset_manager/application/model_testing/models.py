from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ModelType(str, Enum):
    YOLO_DETECTION = "yolo_detection"
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
    overwrite_existing: bool = False


@dataclass(frozen=True, slots=True)
class ModelTestProgress:
    processed: int
    total: int
    succeeded: int
    failed: int
    skipped: int
    current_image: str


@dataclass(frozen=True, slots=True)
class ModelTestResult:
    total: int
    succeeded: int
    failed: int
    generated_annotations: int
    skipped_existing: int
    cancelled: bool = False
    failure_messages: tuple[str, ...] = ()
