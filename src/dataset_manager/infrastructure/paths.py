from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AppPaths:
    root: Path
    config: Path
    projects: Path
    cache: Path
    temp: Path
    resources: Path

    @classmethod
    def discover(cls) -> "AppPaths":
        configured = os.environ.get("DATASET_MANAGER_HOME")
        if configured:
            root = Path(configured).expanduser().resolve()
        elif getattr(sys, "frozen", False):
            root = Path(sys.executable).resolve().parent
        else:
            root = Path.cwd().resolve() / "runtime"
        return cls(
            root=root,
            config=root / "config",
            projects=root / "Projects",
            cache=root / "Cache",
            temp=root / "Temp",
            resources=root / "Resources",
        )

    def ensure_directories(self) -> None:
        for directory in (self.root, self.config, self.projects, self.cache, self.temp, self.resources):
            directory.mkdir(parents=True, exist_ok=True)

