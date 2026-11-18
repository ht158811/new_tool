from __future__ import annotations

import shutil
from pathlib import Path

from dataset_manager.application.ports.project_repository import ProjectRepository
from dataset_manager.domain.models.project import Project
from dataset_manager.infrastructure.storage.json_store import AtomicJsonStore


class ProjectNotFoundError(KeyError):
    pass


class JsonProjectRepository(ProjectRepository):
    def __init__(self, projects_root: Path, store: AtomicJsonStore | None = None) -> None:
        self._root = projects_root.resolve()
        self._store = store or AtomicJsonStore()
        self._root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[Project]:
        projects: list[Project] = []
        for path in sorted(self._root.glob("*/project.json")):
            projects.append(Project.from_dict(self._store.read(path)))
        return sorted(projects, key=lambda project: project.updated_at, reverse=True)

    def get(self, project_id: str) -> Project:
        path = self._project_file(project_id)
        if not path.is_file():
            raise ProjectNotFoundError(project_id)
        return Project.from_dict(self._store.read(path))

    def save(self, project: Project) -> None:
        self._store.write(self._project_file(project.id), project.to_dict())

    def delete(self, project_id: str) -> None:
        directory = self._project_directory(project_id)
        if not (directory / "project.json").is_file():
            raise ProjectNotFoundError(project_id)
        shutil.rmtree(directory)

    def _project_directory(self, project_id: str) -> Path:
        if not project_id or any(character not in "0123456789abcdef" for character in project_id.lower()):
            raise ValueError("无效的项目 ID")
        directory = (self._root / project_id).resolve()
        if directory.parent != self._root:
            raise ValueError("项目路径越界")
        return directory

    def _project_file(self, project_id: str) -> Path:
        return self._project_directory(project_id) / "project.json"

