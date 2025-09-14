"""
CAD generation API endpoints.
Handles CAD model creation from processed text data.
"""

from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
import structlog
from datetime import datetime

from app.core.config import settings
from app.services.cad.cad_service import CADService
from app.schemas.cad import (
    CADGenerationRequest,
    CADGenerationResponse,
    CADProjectResponse,
    GeometryValidationResponse
)
from app.api.deps import get_cad_service, get_current_user
from app.models.project import Project
from app.db.session import get_db

router = APIRouter()
logger = structlog.get_logger()


@router.post("/generate", response_model=CADGenerationResponse)
async def generate_cad(
    request: CADGenerationRequest,
    background_tasks: BackgroundTasks,
    cad_service: CADService = Depends(get_cad_service),
    db = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Generate CAD model from structured input.

    This endpoint:
    1. Accepts processed NLP data or direct CAD instructions
    2. Generates 3D geometry using OpenCASCADE
    3. Applies materials and textures
    4. Validates against building codes
    5. Returns CAD model with metadata

    Complexity levels:
    - L0: Basic shapes (rectangles, circles, simple extrusions)
    - L1: Residential buildings (multi-room, standard materials)
    - L2: Commercial buildings (complex systems, advanced materials)
    - L3: Iconic structures (custom algorithms, parametric design)

    Example request:
    ```json
    {
        "project_name": "Modern Office Building",
        "complexity_level": 2,
        "operations": [
            {
                "type": "create_floor_plan",
                "dimensions": {"length": 50, "width": 30, "unit": "meters"},
                "rooms": [
                    {"type": "office", "count": 10, "area": 20},
                    {"type": "conference", "count": 2, "area": 40}
                ]
            },
            {
                "type": "add_floors",
                "count": 5,
                "height": 3.5
            }
        ],
        "materials": {
            "structure": "steel",
            "facade": "glass",
            "interior": "drywall"
        },
        "constraints": {
            "max_height": 20,
            "setback": 5,
            "green_space_ratio": 0.2
        }
    }
    ```

    Example response:
    ```json
    {
        "project_id": "550e8400-e29b-41d4-a716-446655440000",
        "status": "processing",
        "model_url": "/api/v1/cad/project/550e8400-e29b-41d4-a716-446655440000",
        "preview_url": "/api/v1/render/preview/550e8400-e29b-41d4-a716-446655440000",
        "metadata": {
            "vertices": 15420,
            "faces": 8956,
            "volume": 52500,
            "surface_area": 4200,
            "materials_used": ["steel", "glass", "drywall"],
            "building_code_compliant": true
        },
        "processing_time_ms": 2340,
        "estimated_completion": "2025-09-14T10:30:00Z"
    }
    ```
    """
    try:
        # Create project in database
        project = Project(
            id=uuid4(),
            name=request.project_name,
            user_id=current_user.id if current_user else None,
            complexity_level=request.complexity_level,
            status="processing",
            created_at=datetime.utcnow()
        )
        db.add(project)
        await db.commit()

        # Start CAD generation in background
        background_tasks.add_task(
            cad_service.generate_model,
            project_id=project.id,
            request=request
        )

        logger.info(
            "CAD generation started",
            project_id=str(project.id),
            complexity=request.complexity_level
        )

        return CADGenerationResponse(
            project_id=project.id,
            status="processing",
            model_url=f"/api/v1/cad/project/{project.id}",
            preview_url=f"/api/v1/render/preview/{project.id}",
            estimated_completion=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error("CAD generation failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="CAD generation failed")


@router.get("/project/{project_id}", response_model=CADProjectResponse)
async def get_cad_project(
    project_id: UUID,
    cad_service: CADService = Depends(get_cad_service),
    db = Depends(get_db)
):
    """
    Get CAD project details and download links.

    Returns project status, metadata, and available export formats.
    """
    try:
        project = await db.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Get CAD model data
        model_data = await cad_service.get_model(project_id)

        return CADProjectResponse(
            project_id=project_id,
            name=project.name,
            status=project.status,
            created_at=project.created_at,
            updated_at=project.updated_at,
            model_data=model_data,
            available_formats=settings.SUPPORTED_EXPORT_FORMATS,
            download_links={
                format: f"/api/v1/export/{project_id}?format={format}"
                for format in settings.SUPPORTED_EXPORT_FORMATS
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get project", project_id=str(project_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve project")


@router.post("/validate", response_model=GeometryValidationResponse)
async def validate_geometry(
    file: UploadFile = File(...),
    cad_service: CADService = Depends(get_cad_service)
):
    """
    Validate uploaded CAD geometry.

    Checks:
    - Geometric validity (manifold, watertight)
    - Topology consistency
    - Building code compliance
    - Structural feasibility

    Supported formats: STEP, IFC, STL, OBJ
    """
    try:
        # Read file content
        content = await file.read()

        # Validate geometry
        validation_result = await cad_service.validate_geometry(
            content=content,
            filename=file.filename,
            content_type=file.content_type
        )

        return GeometryValidationResponse(**validation_result)

    except Exception as e:
        logger.error("Geometry validation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Geometry validation failed")


@router.post("/modify/{project_id}")
async def modify_cad_model(
    project_id: UUID,
    modifications: List[Dict[str, Any]],
    cad_service: CADService = Depends(get_cad_service),
    db = Depends(get_db)
):
    """
    Modify existing CAD model.

    Supported modifications:
    - ADD: Add new elements (walls, doors, windows)
    - REMOVE: Remove elements
    - TRANSFORM: Move, rotate, scale elements
    - MATERIAL: Change material properties
    - BOOLEAN: Boolean operations (union, difference, intersection)

    Example:
    ```json
    {
        "modifications": [
            {
                "type": "ADD",
                "element": "window",
                "location": {"wall": "north", "position": {"x": 5, "y": 1.5}},
                "dimensions": {"width": 1.2, "height": 1.5}
            },
            {
                "type": "MATERIAL",
                "target": "exterior_walls",
                "material": "brick"
            }
        ]
    }
    ```
    """
    try:
        project = await db.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Apply modifications
        result = await cad_service.modify_model(
            project_id=project_id,
            modifications=modifications
        )

        # Update project
        project.status = "modified"
        project.updated_at = datetime.utcnow()
        await db.commit()

        return JSONResponse(content={
            "success": True,
            "project_id": str(project_id),
            "modifications_applied": len(modifications),
            "result": result
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Model modification failed", error=str(e))
        raise HTTPException(status_code=500, detail="Model modification failed")


@router.post("/parametric")
async def create_parametric_model(
    parameters: Dict[str, Any],
    template: str = "basic_building",
    cad_service: CADService = Depends(get_cad_service)
):
    """
    Create parametric CAD model from template.

    Templates:
    - basic_building: Simple rectangular building
    - residential_house: Standard house with rooms
    - office_tower: Multi-story office building
    - warehouse: Industrial warehouse structure
    - custom: User-defined parametric model

    Parameters vary by template but typically include:
    - dimensions (length, width, height)
    - floor_count
    - room_layout
    - structural_system
    - facade_type
    """
    try:
        model = await cad_service.create_parametric_model(
            template=template,
            parameters=parameters
        )

        return JSONResponse(content={
            "success": True,
            "model": model,
            "template": template,
            "parameters": parameters
        })

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Parametric model creation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Parametric model creation failed")


@router.get("/templates")
async def list_templates(
    cad_service: CADService = Depends(get_cad_service)
):
    """
    List available parametric templates with their parameters.
    """
    templates = await cad_service.get_available_templates()
    return JSONResponse(content={"templates": templates})