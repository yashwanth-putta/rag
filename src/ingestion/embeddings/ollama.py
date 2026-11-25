from src.ingestion.embeddings.base import BaseEmbeddingModel
from src.config.settings import Settings
from ollama import AsyncClient

class OllamaEmbeddings(BaseEmbeddingModel):
    def __init__(self) -> None:
        self.settings = Settings()
        self.client = AsyncClient() # type: ignore[operator]

    async def get_embedding(self, chunk:str) -> list[float]:
        response = await self.client.embed(
            model = self.settings.OLLAMA_EMBEDDING_MODEL,
            input = chunk
        )

        return response['embeddings'][0]
