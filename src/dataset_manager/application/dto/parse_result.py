from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from dataset_manager.domain.models.labelme import LabelMeDocument


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ParseIssue:
    code: str
    message: str
    severity: Severity
    location: str | None = None


@dataclass(slots=True)
class ParseResult:
    document: LabelMeDocument | None
    issues: list[ParseIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.document is not None and not any(
            issue.severity is Severity.ERROR for issue in self.issues
        )

