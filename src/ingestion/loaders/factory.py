from pathlib import Path
from src.ingestion.loaders.text import TextLoader
from src.ingestion.loaders.base import BaseDocumentLoader

class DocumentLoaderFactory:

    @staticmethod
    def get_loader(file_path: str) -> BaseDocumentLoader:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".txt":
            return TextLoader()

        raise ValueError(f"Unsupported file extension: {ext}")


