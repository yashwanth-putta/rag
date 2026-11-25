from src.ingestion.embeddings.ollama import OllamaEmbeddings
import logging
from src.config.settings import Settings
import os
import numpy as np
import json
from ollama import AsyncClient
from pathlib import Path

logger = logging.getLogger(__name__)

class InferencePipeline:
    def __init__(self):
        self.settings = Settings()

    def build_rag_prompt(self, query: str, chunks: list[str]) -> str:
        if not chunks:
            context_block = "No relevant context was retrieved."
        else:
            formatted_chunks = [
                f"[Chunk {idx}]\n{chunk.strip()}"
                for idx, chunk in enumerate(chunks, start=1)
            ]
            context_block = "\n\n".join(formatted_chunks)

        return (
            "Answer the user's question using ONLY the retrieved context below. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question:\n{query.strip()}"
        )
    async def run(self, query):
        # 1. Embed the Query
        model = OllamaEmbeddings()
        query_embedding = await model.get_embedding(query)
        
        # 2. Similarity search
        data={}
        # 2.1 Load the data. First check DB file exists or not
        vector_dir = self.settings.VECTOR_STORE_DIRECTORY
        store_path = Path(vector_dir) 
        if store_path.exists():
            json_files = [store_path] if store_path.is_file() else sorted(store_path.glob("*.json"))
            for json_file in json_files:
                with open(json_file, "r", encoding="utf-8") as f:
                    try:
                        loaded = json.load(f)
                        if isinstance(loaded, dict):
                            data.update(loaded)
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON in %s", json_file)
        results = []

        # 2.2 Search
        for file_name, file_data in data.items():
            for chunk_data in file_data.get("chunks",[]):
                chunk_embedding = chunk_data.get("embedding")
                # Check the dimensions of fetched data and query embeddings
                if chunk_embedding and len(chunk_embedding) == len(query_embedding):
                    # Similarity Search algorithm
                    # dot product
                    dot_product = np.dot(chunk_embedding,query_embedding)
                    q_norm = np.linalg.norm(query_embedding)
                    db_norms = np.linalg.norm(chunk_embedding)

                    # Cosine Similarity
                    similarity_scores = dot_product / (db_norms * q_norm)
                    results.append(
                        {
                            "chunk": chunk_data["chunk"],
                            "score": similarity_scores,
                            "filename": file_name,
                        }
                    )

        results.sort(key=lambda x:x['score'], reverse=True)
        top_results = results[: self.settings.TOP_K]
        logger.info("%s", top_results)
        context = [item["chunk"] for item in top_results]

        # 3 Query + Chunks ==> LLM
        client = AsyncClient()
        response = await client.chat(
            model = self.settings.OLLAMA_LLM,
            messages= [{
                "role":"user",
                "content": self.build_rag_prompt(query, context)
            }]
        )
        logger.info(response["message"]["content"])

