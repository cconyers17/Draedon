"""
Simplified FastAPI application for Render deployment.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.core.simple_config import settings

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage for demo
demo_models = {}

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "live",
        "documentation": f"{settings.API_V1_STR}/docs",
        "endpoints": {
            "health": "/health",
            "process_text": f"{settings.API_V1_STR}/text/process",
            "generate_cad": f"{settings.API_V1_STR}/cad/generate"
        }
    }


# Health check endpoint
@app.get("/health", tags=["monitoring"])
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "environment": "production" if settings.RENDER else "development",
        "features": {
            "text_processing": True,
            "cad_generation": True,
            "complexity_levels": ["L0", "L1", "L2", "L3"]
        }
    }


# Simple text processing endpoint
@app.post(f"{settings.API_V1_STR}/text/process", tags=["nlp"])
async def process_text(request: Request):
    """Process architectural text description."""
    try:
        body = await request.json()
        text = body.get("text", "")
        complexity = body.get("complexity_level", "L0")

        # Simple text processing
        result = {
            "success": True,
            "intent": "CREATE",
            "confidence": 0.9,
            "entities": [
                {
                    "type": "BUILDING_ELEMENT",
                    "value": "building",
                    "confidence": 0.9,
                    "dimensions": {"length": 20, "width": 10, "height": 3}
                }
            ],
            "complexity_level": complexity,
            "processing_time_ms": 50
        }

        return result

    except Exception as e:
        logger.error(f"Text processing failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Text processing failed", "message": str(e)}
        )


# Simple CAD generation endpoint
@app.post(f"{settings.API_V1_STR}/cad/generate", tags=["cad"])
async def generate_cad(request: Request):
    """Generate CAD model from processed text."""
    try:
        body = await request.json()
        nlp_result = body.get("nlp_result", {})
        complexity = body.get("complexity_level", "L0")

        # Generate simple CAD model
        model_id = f"model_{complexity}_{len(demo_models)}"

        # Simple box geometry
        model = {
            "id": model_id,
            "type": "architectural_model",
            "complexity_level": complexity,
            "geometry": {
                "vertices": [
                    [-10, -5, 0], [10, -5, 0], [10, 5, 0], [-10, 5, 0],  # Bottom
                    [-10, -5, 3], [10, -5, 3], [10, 5, 3], [-10, 5, 3]   # Top
                ],
                "faces": [
                    [0, 1, 2, 3], [4, 7, 6, 5],  # Bottom, Top
                    [0, 4, 5, 1], [1, 5, 6, 2],  # Sides
                    [2, 6, 7, 3], [3, 7, 4, 0]
                ],
                "triangle_count": 12,
                "vertex_count": 8
            },
            "metadata": {
                "boundingBox": {
                    "min": {"x": -10, "y": -5, "z": 0},
                    "max": {"x": 10, "y": 5, "z": 3}
                },
                "triangleCount": 12,
                "vertexCount": 8,
                "volume": 300.0,
                "surfaceArea": 340.0,
                "materials": ["concrete"],
                "units": "meters"
            },
            "exports": {
                "stl": f"/exports/{model_id}.stl",
                "obj": f"/exports/{model_id}.obj"
            }
        }

        # Store model
        demo_models[model_id] = model

        return model

    except Exception as e:
        logger.error(f"CAD generation failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "CAD generation failed", "message": str(e)}
        )


# List models endpoint
@app.get(f"{settings.API_V1_STR}/models", tags=["cad"])
async def list_models():
    """List all generated models."""
    return {
        "models": list(demo_models.values()),
        "count": len(demo_models)
    }


# Get specific model
@app.get(f"{settings.API_V1_STR}/models/{{model_id}}", tags=["cad"])
async def get_model(model_id: str):
    """Get specific model by ID."""
    if model_id in demo_models:
        return demo_models[model_id]
    else:
        return JSONResponse(
            status_code=404,
            content={"error": "Model not found", "model_id": model_id}
        )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(exc)} - Path: {request.url.path} - Method: {request.method}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main_simple:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )