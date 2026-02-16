from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str = ".env", *, override: bool = False) -> bool:
    env_path = Path(path)
    if not env_path.exists() or not env_path.is_file():
        return False

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if not override and key in os.environ:
            continue
        os.environ[key] = value
    return True
