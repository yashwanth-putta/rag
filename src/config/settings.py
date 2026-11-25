from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr
from pathlib import Path

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    app_env: str = "development"
    log_level: str = "INFO"
    chunk_size: int = Field(default=1000, gt=0)
    data_dir: Path = Path("./data")
    api_key: SecretStr | None = None
    OLLAMA_EMBEDDING_MODEL: str = "qwen3-embedding:0.6b"
    VECTOR_STORE_DIRECTORY: str = "embeddings-store"
    TOP_K: int = Field(default=3, gt=0)
    OLLAMA_LLM: str = "qwen2.5:7b"
    GROUND_TRUTH_LOCATION: str = "ground-truth/rag-basics-ground.json"
    OLLAMA_JUDGE_MODEL: str = "qwen3:14b"
