"""Configuration, read from environment variables (a local `.env` file or the host's dashboard).

Nothing sensitive is hard-coded. Paths default to locations relative to the project folder,
so the app finds `data/index.json` no matter where it is started from.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM (answer generation) ---
    gemini_api_key: str | None = None
    llm_model: str = "gemini-2.5-flash"   # check the current free model name in AI Studio
    llm_max_tokens: int = 1500
    llm_temperature: float | None = None

    # --- Embeddings (Gemini API) ---
    gemini_api_key: str | None = None
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768

    # --- The vector index: a JSON file built by scripts/ingest.py and deployed with the app ---
    index_path: str = str(ROOT / "data" / "index.json")
    extra_facts_path: str = str(ROOT / "knowledge" / "extra_facts.md")

    # --- Only used by scripts/ingest.py on your laptop (reads Firestore) ---
    gcp_project_id: str = "portfolio-5bba1"
    firestore_database: str = "(default)"

    # --- Retrieval ---
    top_k: int = 6
    max_distance: float = 0.55  # cosine distance; bigger = more permissive

    # --- Persona ---
    owner_name: str = "Khadija"
    owner_full_name: str = "Khadija Chkhichkh"

    # --- Web security ---
    allowed_origins: str = ""  # comma separated, e.g. https://you.vercel.app
    allowed_origin_regex: str | None = None
    rate_limit_per_minute: int = 10
    client_ip_hops: int = 1  # which X-Forwarded-For entry, counted from the right, is the real client
    enable_docs: bool = False

    @property
    def origins(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
