"""
Application configuration using Pydantic Settings.
Handles environment variables and configuration management.
"""

from typing import Optional, List, Any, Dict
# from pydantic_settings import BaseSettings
# from pydantic import PostgresDsn, RedisDsn, validator, Field
import secrets
import os
from pathlib import Path


class Settings:
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Text-to-CAD API"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False)

    # Security
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 1 week
    ALGORITHM: str = "HS256"

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"]
    )

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Database
    POSTGRES_SERVER: str = Field(default="localhost")
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="postgres")
    POSTGRES_DB: str = Field(default="text_to_cad")
    DATABASE_URL: Optional[PostgresDsn] = None

    @validator("DATABASE_URL", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            username=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_SERVER"),
            path=f"{values.get('POSTGRES_DB') or ''}",
        )

    # Redis Cache
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_DB: int = Field(default=0)
    REDIS_PASSWORD: Optional[str] = Field(default=None)
    REDIS_URL: Optional[RedisDsn] = None

    @validator("REDIS_URL", pre=True)
    def assemble_redis_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        password = values.get("REDIS_PASSWORD")
        return RedisDsn.build(
            scheme="redis",
            username="" if not password else None,
            password=password,
            host=values.get("REDIS_HOST"),
            port=str(values.get("REDIS_PORT")),
            path=f"/{values.get('REDIS_DB') or 0}",
        )

    # File Storage
    USE_S3: bool = Field(default=False)
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None)
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None)
    AWS_REGION: str = Field(default="us-east-1")
    S3_BUCKET_NAME: str = Field(default="text-to-cad-files")
    LOCAL_STORAGE_PATH: Path = Field(default=Path("/tmp/text-to-cad"))
    MAX_UPLOAD_SIZE: int = Field(default=100 * 1024 * 1024)  # 100MB

    # External APIs
    MESHY_API_KEY: Optional[str] = Field(default=None)
    MESHY_API_URL: str = Field(default="https://api.meshy.ai")
    MESHY_FREE_LIMIT: int = Field(default=200)  # Monthly limit

    TRELLIS_API_KEY: Optional[str] = Field(default=None)
    TRELLIS_API_URL: str = Field(default="https://api.trellis3d.com")

    RODIN_API_KEY: Optional[str] = Field(default=None)
    RODIN_API_URL: str = Field(default="https://api.rodin.ai")

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True)
    RATE_LIMIT_REQUESTS: int = Field(default=100)
    RATE_LIMIT_PERIOD: int = Field(default=60)  # seconds

    # Background Tasks
    USE_CELERY: bool = Field(default=True)
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/1")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/2")

    # NLP Configuration
    SPACY_MODEL: str = Field(default="en_core_web_lg")
    NLP_CACHE_TTL: int = Field(default=3600)  # 1 hour

    # CAD Configuration
    DEFAULT_EXPORT_FORMAT: str = Field(default="STEP")
    SUPPORTED_EXPORT_FORMATS: List[str] = Field(
        default=["STEP", "IFC", "STL", "OBJ", "DXF", "IGES"]
    )
    CAD_PRECISION: float = Field(default=0.001)  # meters
    MAX_COMPLEXITY_LEVEL: int = Field(default=3)  # L0-L3

    # Monitoring
    SENTRY_DSN: Optional[str] = Field(default=None)
    ENABLE_METRICS: bool = Field(default=True)
    METRICS_PORT: int = Field(default=9090)

    # Deployment (Render.com specific)
    PORT: int = Field(default=8000)
    HOST: str = Field(default="0.0.0.0")
    WORKERS: int = Field(default=4)
    RENDER: bool = Field(default=False)  # Set to True when deployed on Render

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()