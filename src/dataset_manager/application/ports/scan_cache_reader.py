from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ScanCacheReader(ABC):
    @abstractmethod
    def read(self, dataset_id: str, scan_id: str) -> dict[str, Any]: ...

