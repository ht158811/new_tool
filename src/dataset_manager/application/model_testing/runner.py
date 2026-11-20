from __future__ import annotations

from typing import Any, Protocol

from dataset_manager.application.model_testing.models import ModelTestConfig


class ModelRunner(Protocol):
    def load(self, config: ModelTestConfig) -> dict[str, Any]: ...

    def predict(self, image_path: str, config: ModelTestConfig) -> dict[str, Any]: ...
