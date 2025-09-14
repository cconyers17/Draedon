"""
Simplified configuration for Render deployment.
"""

import os
import secrets
from typing import List


class Settings:
    """Simple settings class without Pydantic."""

    def __init__(self):
        # API Configuration
        self.API_V1_STR = "/api/v1"
        self.PROJECT_NAME = "Draedon Text-to-CAD API"
        self.VERSION = "1.0.0"
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"

        # Security
        self.SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))

        # CORS
        cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
        self.BACKEND_CORS_ORIGINS = [origin.strip() for origin in cors_origins.split(",")]

        # Server
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = int(os.getenv("PORT", "8000"))
        self.WORKERS = int(os.getenv("WORKERS", "1"))

        # Database (optional for now)
        self.DATABASE_URL = os.getenv("DATABASE_URL")
        self.REDIS_URL = os.getenv("REDIS_URL")

        # NLP
        self.SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
        self.NLP_CACHE_TTL = int(os.getenv("NLP_CACHE_TTL", "3600"))

        # File handling
        self.MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "104857600"))

        # Render detection
        self.RENDER = os.getenv("RENDER", "false").lower() == "true"

        # Monitoring
        self.ENABLE_METRICS = os.getenv("ENABLE_METRICS", "false").lower() == "true"
        self.SENTRY_DSN = os.getenv("SENTRY_DSN")


# Create global settings instance
settings = Settings()