from __future__ import annotations

from pathlib import Path

from dataset_manager.application.dto.scan_result import ScanResult
from dataset_manager.application.ports.scan_cache_reader import ScanCacheReader
from dataset_manager.infrastructure.storage.json_store import AtomicJsonStore


class ScanCacheStore(ScanCacheReader):
    """Stores immutable scan snapshots addressed by dataset ID and scan ID."""

    def __init__(self, cache_root: Path, json_store: AtomicJsonStore | None = None) -> None:
        self._root = cache_root.resolve() / "scans"
        self._store = json_store or AtomicJsonStore()
        self._root.mkdir(parents=True, exist_ok=True)

    def commit(self, result: ScanResult) -> Path:
        path = self.path_for(result.dataset_id, result.scan_id)
        self._store.write(path, result.to_dict())
        return path

    def read(self, dataset_id: str, scan_id: str) -> dict[str, object]:
        return self._store.read(self.path_for(dataset_id, scan_id))

    def delete(self, dataset_id: str, scan_id: str) -> None:
        self.path_for(dataset_id, scan_id).unlink(missing_ok=True)

    def path_for(self, dataset_id: str, scan_id: str) -> Path:
        self._validate_id(dataset_id)
        self._validate_id(scan_id)
        directory = (self._root / dataset_id).resolve()
        if directory.parent != self._root:
            raise ValueError("缓存路径越界")
        return directory / f"{scan_id}.json"

    @staticmethod
    def _validate_id(value: str) -> None:
        if not value or any(character not in "0123456789abcdef" for character in value.casefold()):
            raise ValueError("无效的缓存标识")

