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
class Dataset:
    id: str
    name: str
    root_path: str
    image_directory: str
    annotation_directory: str
    task_type: TaskType
    data_format: str = "labelme"
    accessible: bool = True


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
            dataset["task_type"] = dataset["task_type"].value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Project":
        datasets = [
            Dataset(**{**item, "task_type": TaskType(item["task_type"])})
            for item in data.get("datasets", [])
        ]
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            datasets=datasets,
            schema_version=int(data.get("schema_version", 1)),
        )

