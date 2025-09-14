"""
File export API endpoints.
Handles CAD model export in various formats.
"""

from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
import structlog
from pathlib import Path
import io

from app.core.config import settings
from app.services.export.export_service import ExportService
from app.api.deps import get_export_service, get_current_user
from app.db.session import get_db
from app.models.project import Project

router = APIRouter()
logger = structlog.get_logger()


@router.get("/{project_id}")
async def export_cad_model(
    project_id: UUID,
    format: str = Query(
        default=settings.DEFAULT_EXPORT_FORMAT,
        description="Export format",
        regex="^(STEP|IFC|STL|OBJ|DXF|IGES)$"
    ),
    quality: str = Query(
        default="standard",
        description="Export quality",
        regex="^(draft|standard|high)$"
    ),
    units: str = Query(
        default="meters",
        description="Unit system",
        regex="^(meters|millimeters|feet|inches)$"
    ),
    export_service: ExportService = Depends(get_export_service),
    db = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Export CAD model in specified format.

    Supported formats:
    - **STEP**: ISO 10303 standard, preserves full geometry and topology
    - **IFC**: Building Information Modeling, includes metadata and properties
    - **STL**: 3D printing, triangulated surface mesh
    - **OBJ**: Wavefront, includes materials and textures
    - **DXF**: AutoCAD exchange format, 2D/3D drawings
    - **IGES**: Legacy CAD exchange format

    Quality levels:
    - **draft**: Fast export, lower precision (0.01m tolerance)
    - **standard**: Balanced quality/speed (0.001m tolerance)
    - **high**: Maximum precision (0.0001m tolerance)

    Example usage:
    ```
    GET /api/v1/export/550e8400-e29b-41d4-a716-446655440000?format=STEP&quality=high&units=millimeters
    ```

    Returns file download with appropriate MIME type.
    """
    try:
        # Verify project exists and user has access
        project = await db.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        if current_user and project.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")

        # Export model
        export_result = await export_service.export_model(
            project_id=project_id,
            format=format,
            quality=quality,
            units=units
        )

        # Determine MIME type
        mime_types = {
            "STEP": "application/step",
            "IFC": "application/x-ifc",
            "STL": "application/vnd.ms-pki.stl",
            "OBJ": "text/plain",
            "DXF": "application/dxf",
            "IGES": "application/iges"
        }

        filename = f"{project.name}_{project_id}.{format.lower()}"

        logger.info(
            "Model exported",
            project_id=str(project_id),
            format=format,
            quality=quality,
            size_bytes=len(export_result.content)
        )

        return StreamingResponse(
            io.BytesIO(export_result.content),
            media_type=mime_types.get(format, "application/octet-stream"),
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(export_result.content)),
                "X-Export-Quality": quality,
                "X-Export-Units": units
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Export failed",
            project_id=str(project_id),
            format=format,
            error=str(e)
        )
        raise HTTPException(status_code=500, detail="Export failed")


@router.post("/{project_id}/batch")
async def batch_export(
    project_id: UUID,
    formats: list[str],
    quality: str = "standard",
    export_service: ExportService = Depends(get_export_service),
    db = Depends(get_db)
):
    """
    Export CAD model in multiple formats simultaneously.

    Creates a ZIP archive containing the model in all requested formats.

    Example request:
    ```json
    {
        "formats": ["STEP", "STL", "OBJ"],
        "quality": "high"
    }
    ```

    Returns ZIP file download.
    """
    try:
        # Validate formats
        invalid_formats = [f for f in formats if f not in settings.SUPPORTED_EXPORT_FORMATS]
        if invalid_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported formats: {', '.join(invalid_formats)}"
            )

        # Verify project exists
        project = await db.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Perform batch export
        zip_result = await export_service.batch_export(
            project_id=project_id,
            formats=formats,
            quality=quality
        )

        filename = f"{project.name}_{project_id}_batch.zip"

        return StreamingResponse(
            io.BytesIO(zip_result),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(zip_result))
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch export failed", error=str(e))
        raise HTTPException(status_code=500, detail="Batch export failed")


@router.get("/{project_id}/2d-drawings")
async def export_2d_drawings(
    project_id: UUID,
    views: list[str] = Query(
        default=["top", "front", "side"],
        description="Drawing views to generate"
    ),
    scale: str = Query(
        default="1:100",
        description="Drawing scale"
    ),
    format: str = Query(
        default="DXF",
        description="2D drawing format",
        regex="^(DXF|SVG|PDF)$"
    ),
    export_service: ExportService = Depends(get_export_service),
    db = Depends(get_db)
):
    """
    Generate 2D technical drawings from 3D model.

    Views:
    - top: Plan view
    - front: Front elevation
    - side: Side elevation
    - section: Cross-section
    - isometric: 3D isometric view

    Scales: 1:50, 1:100, 1:200, 1:500

    Formats:
    - DXF: AutoCAD format
    - SVG: Vector graphics
    - PDF: Print-ready document
    """
    try:
        project = await db.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        drawings = await export_service.generate_2d_drawings(
            project_id=project_id,
            views=views,
            scale=scale,
            format=format
        )

        filename = f"{project.name}_drawings.{format.lower()}"

        return StreamingResponse(
            io.BytesIO(drawings),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(drawings))
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("2D drawing export failed", error=str(e))
        raise HTTPException(status_code=500, detail="2D drawing export failed")


@router.get("/{project_id}/bom")
async def export_bill_of_materials(
    project_id: UUID,
    format: str = Query(
        default="JSON",
        description="BOM format",
        regex="^(JSON|CSV|XLSX|PDF)$"
    ),
    include_costs: bool = Query(
        default=False,
        description="Include cost estimates"
    ),
    export_service: ExportService = Depends(get_export_service),
    db = Depends(get_db)
):
    """
    Export Bill of Materials (BOM) for the CAD model.

    Includes:
    - Material quantities
    - Component list
    - Dimensions and specifications
    - Optional cost estimates
    - Carbon footprint data

    Example BOM structure:
    ```json
    {
        "project_id": "550e8400-e29b-41d4-a716-446655440000",
        "materials": [
            {
                "name": "Concrete",
                "quantity": 150,
                "unit": "m³",
                "cost_per_unit": 120,
                "total_cost": 18000,
                "carbon_kg": 450
            }
        ],
        "components": [
            {
                "type": "Door",
                "count": 15,
                "specifications": "Standard interior door 2.1m x 0.9m"
            }
        ],
        "summary": {
            "total_cost": 85000,
            "total_carbon_kg": 12500,
            "total_weight_kg": 450000
        }
    }
    ```
    """
    try:
        project = await db.get(Project, project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        bom = await export_service.generate_bom(
            project_id=project_id,
            format=format,
            include_costs=include_costs
        )

        mime_types = {
            "JSON": "application/json",
            "CSV": "text/csv",
            "XLSX": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "PDF": "application/pdf"
        }

        filename = f"{project.name}_BOM.{format.lower()}"

        return StreamingResponse(
            io.BytesIO(bom),
            media_type=mime_types.get(format, "application/octet-stream"),
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(bom))
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("BOM export failed", error=str(e))
        raise HTTPException(status_code=500, detail="BOM export failed")