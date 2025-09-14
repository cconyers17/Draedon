"""
API v1 router configuration.
Aggregates all endpoint routers.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    text_processing,
    cad_generation,
    export,
    projects,
    rendering,
    webhooks
)

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    text_processing.router,
    prefix="/text",
    tags=["text-processing"]
)

api_router.include_router(
    cad_generation.router,
    prefix="/cad",
    tags=["cad-generation"]
)

api_router.include_router(
    export.router,
    prefix="/export",
    tags=["export"]
)

api_router.include_router(
    projects.router,
    prefix="/projects",
    tags=["projects"]
)

api_router.include_router(
    rendering.router,
    prefix="/render",
    tags=["3d-rendering"]
)

api_router.include_router(
    webhooks.router,
    prefix="/webhooks",
    tags=["webhooks"]
)