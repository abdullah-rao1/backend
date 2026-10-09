from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every setting can be overridden with an environment variable of the same name
    (upper-cased), e.g. GROQ_API_KEY, DATABASE_URL, CHROMA_PATH."""

    app_name: str = "Smart RAG"
    app_env: str = "development"

    # Local disk paths. On Railway these must live on the attached volume
    # (e.g. /app/storage/...) or they are wiped on every redeploy.
    upload_dir: Path = Path("storage/uploads")
    chroma_path: Path = Path("storage/chroma")
    chroma_collection: str = "session_chunks"
    max_file_size_mb: int = 25

    # 384-dimension sentence-transformers model (~90 MB, fast on CPU).
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    groq_api_key: str = ""
    # Fallbacks are tried in order if the primary model is overloaded (503), rate-limited
    # (429) or retired (404). Comma-separated; set to "" to disable.
    # NOTE: Groq retired llama-3.1-8b-instant and llama-3.3-70b-versatile on 2026-08-16.
    groq_model: str = "openai/gpt-oss-20b"
    groq_fallback_models: str = "openai/gpt-oss-120b"
    groq_reasoning_effort: str = "low"  # only used by openai/gpt-oss-* models
    groq_max_retries: int = 2
    groq_timeout_seconds: int = 30

    # Optional: enables the web-search fallback step
    tavily_api_key: str = ""

    cors_origins: str = "*"
    # Optional regex for extra allowed origins, e.g. Vercel preview deployments:
    #   CORS_ORIGIN_REGEX=https://.*\.vercel\.app
    cors_origin_regex: str = ""

    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/smart_rag"

    jwt_secret_key: str = "change-this-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        # Browsers send the Origin header WITHOUT a trailing slash
        # ("https://app.up.railway.app"), and Starlette's CORS check is an exact string
        # match. So "https://app.up.railway.app/" would never match -> every request is
        # blocked. Strip quotes/whitespace/trailing slashes to make the setting forgiving.
        origins = []
        for origin in self.cors_origins.split(","):
            origin = origin.strip().strip('"').strip("'").rstrip("/")
            if origin:
                origins.append(origin)
        return origins or ["*"]

    @property
    def groq_model_list(self) -> list[str]:
        models = [self.groq_model.strip()]
        for name in self.groq_fallback_models.split(","):
            name = name.strip()
            if name and name not in models:
                models.append(name)
        return [m for m in models if m]


def _normalize_database_url(url: str) -> str:
    # Some hosts (Railway, Heroku-style Postgres add-ons) inject the legacy "postgres://"
    # scheme, which SQLAlchemy 1.4+ no longer accepts.
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]
    # SQLAlchemy 2.1 changed the default driver for "postgresql://" from psycopg2 to
    # psycopg (v3). We ship psycopg2-binary, so pin the driver explicitly.
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg2://" + url[len("postgresql://") :]
    return url


def _check_production_safety(settings: "Settings") -> None:
    if settings.app_env != "production":
        return

    problems = []
    if settings.jwt_secret_key == "change-this-secret-key":
        problems.append("JWT_SECRET_KEY is still the placeholder default")
    if not settings.groq_api_key:
        problems.append("GROQ_API_KEY is not set")
    if "localhost" in settings.database_url or "127.0.0.1" in settings.database_url:
        problems.append("DATABASE_URL still points at localhost")

    if problems:
        raise RuntimeError(
            "Refusing to start with APP_ENV=production and unsafe defaults: " + "; ".join(problems)
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.database_url = _normalize_database_url(settings.database_url)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_path.mkdir(parents=True, exist_ok=True)

    _check_production_safety(settings)
    return settings


settings = get_settings()
