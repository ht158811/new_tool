from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataset_manager.application.services.project_service import ProjectService
from dataset_manager.infrastructure.storage.project_repository import JsonProjectRepository


class ProjectServiceTests(unittest.TestCase):
    def test_create_list_and_delete_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ProjectService(JsonProjectRepository(Path(directory)))
            project = service.create_project("PCB")
            self.assertEqual(service.list_projects()[0].name, "PCB")
            service.delete_project_management_info(project.id)
            self.assertEqual(service.list_projects(), [])

    def test_rejects_blank_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = ProjectService(JsonProjectRepository(Path(directory)))
            with self.assertRaises(ValueError):
                service.create_project("   ")


if __name__ == "__main__":
    unittest.main()
