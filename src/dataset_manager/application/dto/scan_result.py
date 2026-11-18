from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ScanIssue:
    code: str
    message: str
    severity: str
    path: str | None = None


@dataclass(slots=True)
class ScanRecord:
    image_path: str | None
    annotation_path: str | None
    width: int | None = None
    height: int | None = None
    categories: list[str] = field(default_factory=list)
    issues: list[ScanIssue] = field(default_factory=list)
    annotation_valid: bool = False


@dataclass(slots=True)
class ScanResult:
    scan_id: str
    dataset_id: str
    completed_at: str
    records: list[ScanRecord]
    issues: list[ScanIssue]
    image_count: int
    annotation_count: int
    paired_count: int
    categories: list[str]
    schema_version: int = 1

    @property
    def issue_count(self) -> int:
        return len(self.issues) + sum(len(record.issues) for record in self.records)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScanProgress:
    current_path: str
    processed: int
    total: int
    issue_count: int

