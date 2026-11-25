import logging
import sys
from src.ingestion.pipeline import DocumentIngestionPipeline
import asyncio

def setup_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    
async def main() -> None:
    setup_logging("INFO")
    ingestion_pipeline = DocumentIngestionPipeline()
    await ingestion_pipeline.ingest_file("knowledge-factory/rag-basics.txt")


if __name__ == "__main__":
    asyncio.run(main())
