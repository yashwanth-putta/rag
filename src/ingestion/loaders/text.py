from src.ingestion.loaders.base import BaseDocumentLoader
import asyncio
import logging

logger = logging.getLogger(__name__)

class TextLoader(BaseDocumentLoader):
    """
    Read the text file and return the content in the form of string back
    """
    async def load(self, file_path) -> str:
        loop = asyncio.get_event_loop()

        def _read_file():
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        """
        This line is telling the event loop to hand off _read_file function to the background worker thread 
        so that it need not wait for the disk and wake me up when it finishes and give me the result.
        Waiting for disk and reading a file is a blocking operation.
        """
        return await loop.run_in_executor(None, _read_file)
