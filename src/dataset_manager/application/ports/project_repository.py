from __future__ import annotations

from abc import ABC, abstractmethod

from dataset_manager.domain.models.project import Project


class ProjectRepository(ABC):
    @abstractmethod
    def list(self) -> list[Project]: ...

    @abstractmethod
    def get(self, project_id: str) -> Project: ...

    @abstractmethod
    def save(self, project: Project) -> None: ...

    @abstractmethod
    def delete(self, project_id: str) -> None: ...

