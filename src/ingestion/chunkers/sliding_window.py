from abc import ABC
from src.ingestion.chunkers.base import BaseChunker

class SlidingWindowChunking(BaseChunker):
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    async def chunk(self, text):
        chunks = []
        start = 0
        text_length = len(text)

        if not text:
            return chunks

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunks.append(text[start:end])
            if end == text_length:
                break
            start += self.chunk_size - self.overlap

        return chunks
