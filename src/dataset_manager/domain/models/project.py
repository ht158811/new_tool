from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskType(str, Enum):
    OBJECT_DETECTION = "object_detection"
    IMAGE_CLASSIFICATION = "image_classification"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"


@dataclass(slots=True)
class ScanSummary:
    image_count: int = 0
    annotation_count: int = 0
    paired_count: int = 0
    valid_annotation_image_count: int = 0
    issue_count: int = 0
    category_count: int = 0
    categories: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Dataset:
    id: str
    name: str
    root_path: str
    image_directory: str
    annotation_directory: str
    task_type: TaskType
    data_format: str = "labelme"
    accessible: bool = True
    active_scan_id: str | None = None
    last_scanned_at: str | None = None
    scan_summary: ScanSummary = field(default_factory=ScanSummary)

    def __post_init__(self) -> None:
        # Qt item data serializes str-based enums as plain strings. Normalize at
        # the domain boundary so scanners and persistence always see TaskType.
        if not isinstance(self.task_type, TaskType):
            self.task_type = TaskType(self.task_type)

    @classmethod
    def create(
        cls,
        name: str,
        root_path: str,
        image_directory: str,
        annotation_directory: str,
        task_type: TaskType | str,
    ) -> "Dataset":
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("数据集名称不能为空")
        return cls(
            id=uuid4().hex,
            name=clean_name,
            root_path=root_path,
            image_directory=image_directory,
            annotation_directory=annotation_directory,
            task_type=task_type,
        )


@dataclass(slots=True)
class Project:
    id: str
    name: str
    created_at: str
    updated_at: str
    datasets: list[Dataset] = field(default_factory=list)
    schema_version: int = 1

    @classmethod
    def create(cls, name: str) -> "Project":
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("项目名称不能为空")
        now = utc_now_iso()
        return cls(id=uuid4().hex, name=clean_name, created_at=now, updated_at=now)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for dataset in data["datasets"]:
            task_type = dataset["task_type"]
            dataset["task_type"] = (
                task_type.value if isinstance(task_type, TaskType) else TaskType(task_type).value
            )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Project":
        datasets = []
        for item in data.get("datasets", []):
            dataset_data = dict(item)
            dataset_data["task_type"] = TaskType(dataset_data["task_type"])
            dataset_data["scan_summary"] = ScanSummary(**dataset_data.get("scan_summary", {}))
            datasets.append(Dataset(**dataset_data))
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            datasets=datasets,
            schema_version=int(data.get("schema_version", 1)),
        )

    def find_dataset(self, dataset_id: str) -> Dataset:
        for dataset in self.datasets:
            if dataset.id == dataset_id:
                return dataset
        raise KeyError(dataset_id)

