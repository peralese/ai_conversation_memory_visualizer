from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseParser(ABC):
    @abstractmethod
    def parse_file(self, file_path: str) -> list[dict[str, Any]]:
        raise NotImplementedError
