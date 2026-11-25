from abc import ABC, abstractmethod
import asyncio
from pathlib import Path

class BaseDocumentLoader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    async def load(self, path) -> str:
        pass
