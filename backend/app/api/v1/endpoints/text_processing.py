"""
Text processing API endpoints.
Handles natural language input parsing and intent extraction.
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings
from app.services.nlp.nlp_service import NLPService
from app.schemas.nlp import (
    TextProcessingRequest,
    TextProcessingResponse,
    EntityExtractionResponse,
    IntentClassificationResponse
)
from app.utils.cache import cache_key_wrapper, get_cache, set_cache
from app.api.deps import get_nlp_service

router = APIRouter()
logger = structlog.get_logger()
limiter = Limiter(key_func=get_remote_address)


@router.post("/process", response_model=TextProcessingResponse)
@limiter.limit(f"{settings.RATE_LIMIT_REQUESTS}/minute")
async def process_text(
    request: Request,
    text_request: TextProcessingRequest,
    nlp_service: NLPService = Depends(get_nlp_service),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Process natural language text to extract CAD generation parameters.

    This endpoint:
    1. Extracts architectural entities (walls, doors, windows, etc.)
    2. Identifies dimensions and spatial relationships
    3. Classifies user intent (CREATE, MODIFY, REMOVE, QUERY)
    4. Validates against building codes
    5. Returns structured data ready for CAD generation

    Example request:
    ```json
    {
        "text": "Create a 10m x 15m office with two windows on the north wall",
        "complexity_level": 1,
        "context": {
            "building_type": "commercial",
            "location": "urban"
        }
    }
    ```

    Example response:
    ```json
    {
        "success": true,
        "intent": "CREATE",
        "entities": [
            {
                "type": "SPACE",
                "value": "office",
                "dimensions": {"length": 10, "width": 15, "unit": "meters"}
            },
            {
                "type": "BUILDING_ELEMENT",
                "value": "window",
                "quantity": 2,
                "location": "north wall"
            }
        ],
        "constraints": [
            {
                "type": "spatial",
                "description": "windows on north wall"
            }
        ],
        "validation": {
            "building_code_compliant": true,
            "warnings": []
        },
        "processing_time_ms": 145
    }
    ```
    """
    try:
        # Check cache
        cache_key = cache_key_wrapper(
            "nlp:process",
            text_request.text,
            text_request.complexity_level
        )
        cached_result = await get_cache(cache_key)
        if cached_result:
            logger.info("Cache hit for text processing", cache_key=cache_key)
            return TextProcessingResponse(**cached_result)

        # Process text
        result = await nlp_service.process_text(
            text=text_request.text,
            complexity_level=text_request.complexity_level,
            context=text_request.context
        )

        # Cache result
        background_tasks.add_task(
            set_cache,
            cache_key,
            result.dict(),
            ttl=settings.NLP_CACHE_TTL
        )

        logger.info(
            "Text processed successfully",
            intent=result.intent,
            entity_count=len(result.entities),
            processing_time=result.processing_time_ms
        )

        return result

    except ValueError as e:
        logger.error("Invalid input for text processing", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Text processing failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Text processing failed")


@router.post("/extract-entities", response_model=EntityExtractionResponse)
async def extract_entities(
    text: str = Field(..., min_length=3, max_length=5000),
    nlp_service: NLPService = Depends(get_nlp_service)
):
    """
    Extract architectural entities from text.

    Identifies:
    - Building elements (walls, doors, windows, roofs, etc.)
    - Dimensions and measurements
    - Materials
    - Spatial relationships
    - Quantities

    Example:
    Input: "Build a concrete wall 5 meters high with a steel door"
    Output:
    ```json
    {
        "entities": [
            {"type": "MATERIAL", "value": "concrete", "confidence": 0.95},
            {"type": "BUILDING_ELEMENT", "value": "wall", "confidence": 0.98},
            {"type": "DIMENSION", "value": "5 meters", "parsed": {"value": 5, "unit": "meters"}},
            {"type": "MATERIAL", "value": "steel", "confidence": 0.93},
            {"type": "BUILDING_ELEMENT", "value": "door", "confidence": 0.97}
        ]
    }
    ```
    """
    try:
        entities = await nlp_service.extract_entities(text)
        return EntityExtractionResponse(entities=entities)
    except Exception as e:
        logger.error("Entity extraction failed", error=str(e))
        raise HTTPException(status_code=500, detail="Entity extraction failed")


@router.post("/classify-intent", response_model=IntentClassificationResponse)
async def classify_intent(
    text: str = Field(..., min_length=3, max_length=5000),
    nlp_service: NLPService = Depends(get_nlp_service)
):
    """
    Classify user intent from text.

    Intent types:
    - CREATE: Generate new CAD elements
    - MODIFY: Change existing elements
    - REMOVE: Delete elements
    - QUERY: Ask about elements or get information

    Example:
    Input: "Add a window to the east wall"
    Output:
    ```json
    {
        "intent": "CREATE",
        "confidence": 0.92,
        "sub_intent": "add_building_element",
        "target": "window",
        "location": "east wall"
    }
    ```
    """
    try:
        intent_result = await nlp_service.classify_intent(text)
        return IntentClassificationResponse(**intent_result)
    except Exception as e:
        logger.error("Intent classification failed", error=str(e))
        raise HTTPException(status_code=500, detail="Intent classification failed")


@router.post("/validate-constraints")
async def validate_constraints(
    entities: List[Dict[str, Any]],
    building_type: str = "residential",
    nlp_service: NLPService = Depends(get_nlp_service)
):
    """
    Validate extracted entities against building codes and constraints.

    Checks:
    - Minimum room dimensions
    - Door/window size limits
    - Structural requirements
    - Building code compliance
    - Material compatibility

    Returns validation results with any warnings or errors.
    """
    try:
        validation_result = await nlp_service.validate_constraints(
            entities=entities,
            building_type=building_type
        )
        return validation_result
    except Exception as e:
        logger.error("Constraint validation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Constraint validation failed")


@router.post("/generate-prompt")
async def generate_cad_prompt(
    processed_data: TextProcessingResponse,
    target_format: str = "parametric"
):
    """
    Generate a structured prompt for CAD generation from processed text.

    Converts NLP output to CAD-ready instructions including:
    - Geometric primitives
    - Dimensional constraints
    - Material specifications
    - Assembly instructions

    Formats:
    - parametric: Parametric modeling commands
    - csg: Constructive Solid Geometry operations
    - script: CAD scripting language
    """
    try:
        # Convert processed NLP data to CAD generation prompt
        cad_prompt = {
            "format": target_format,
            "operations": [],
            "constraints": [],
            "materials": []
        }

        for entity in processed_data.entities:
            if entity.type == "BUILDING_ELEMENT":
                operation = {
                    "type": "create",
                    "element": entity.value,
                    "parameters": entity.get("dimensions", {})
                }
                cad_prompt["operations"].append(operation)

        return JSONResponse(content=cad_prompt)

    except Exception as e:
        logger.error("CAD prompt generation failed", error=str(e))
        raise HTTPException(status_code=500, detail="CAD prompt generation failed")