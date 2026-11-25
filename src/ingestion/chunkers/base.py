from abc import abstractmethod

class BaseChunker:
    @abstractmethod
    async def chunk(self, text:str) -> list[str]:
        pass
