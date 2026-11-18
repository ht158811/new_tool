from __future__ import annotations

from dataset_manager.application.ports.project_repository import ProjectRepository
from dataset_manager.domain.models.project import Project


class ProjectService:
    def __init__(self, repository: ProjectRepository) -> None:
        self._repository = repository

    def list_projects(self) -> list[Project]:
        return self._repository.list()

    def create_project(self, name: str) -> Project:
        project = Project.create(name)
        self._repository.save(project)
        return project

    def delete_project_management_info(self, project_id: str) -> None:
        """Delete only app metadata; never touch external dataset directories."""
        self._repository.get(project_id)
        self._repository.delete(project_id)

