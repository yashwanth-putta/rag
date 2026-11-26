from abc import ABC, abstractmethod

class BaseEmbeddingModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    async def get_embedding(self, chunk:str) -> str:
        pass
