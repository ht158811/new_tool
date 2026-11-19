from __future__ import annotations

from abc import ABC, abstractmethod


class ScanCacheManager(ABC):
    @abstractmethod
    def delete(self, dataset_id: str, scan_id: str) -> None: ...

