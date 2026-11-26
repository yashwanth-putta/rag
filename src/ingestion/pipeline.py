from src.ingestion.loaders.factory import DocumentLoaderFactory
from src.ingestion.chunkers.sliding_window import SlidingWindowChunking
from src.ingestion.embeddings.ollama import OllamaEmbeddings
import logging
import os
from src.config.settings import Settings
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class DocumentIngestionPipeline:
    def __init__(self):
        self.settings = Settings()


    async def ingest_file(self, file_path) -> str:
        path_obj = Path(file_path)
        file_name = path_obj.name
        file_type = path_obj.suffix.lstrip(".")
            
        # 1. Load Documents
        logger.info("[1] Loading documents...")
        loader = DocumentLoaderFactory.get_loader(file_path) 
        content = await loader.load(file_path)

        # 2. Chunk
        logger.info("[2] Chunking.....")
        chunker = SlidingWindowChunking()
        chunks = await chunker.chunk(content)
    
        # 3. Embed
        logger.info("[3] Embedding....")
        total = len(chunks)
        vectors = []
        model = OllamaEmbeddings()
        for idx, chunk in enumerate(chunks, start=1):
            logger.info("Embedding chunk %d/%d (length=%d)", idx, total, len(chunk))
            embedding = await model.get_embedding(chunk)
            vectors.append(embedding)

        # 4. Store  
         
        logger.info("[4] Storing embeddings...")
        store_dir = Path(self.settings.VECTOR_STORE_DIRECTORY)
        store_dir.mkdir(parents=True, exist_ok=True)
        output_path = store_dir / f"{path_obj.stem}.json"

        data: dict = {}
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}

        records = [
            {"chunk": c, "embedding": e}
            for c, e in zip(chunks, vectors)
        ]

        data[file_name] = {
            "file_name": file_name,
            "file_type": file_type,
            "chunks": records,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        logger.info("Saved %d chunks to %s", len(records), output_path)
        return str(output_path)


    


