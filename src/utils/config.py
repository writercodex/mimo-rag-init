from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    MIMO_API_KEY: str = "your_mimo_api_key_here"
    MIMO_API_BASE_URL: str = "https://api.platform.xiaomimimo.com/v1"
    MIMO_MODEL: str = "mimo-vl-7b-rl"
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    MAX_RETRIEVED_CHUNKS: int = 5
    VECTOR_STORE: str = "faiss"
    INDEX_PATH: str = "./data/index/mimo_rag"
    DOCUMENTS_DIR: str = "./data/documents"

settings = Settings()
