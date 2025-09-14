"""
CAD Service for Text-to-CAD Architecture Application.
Handles 3D model generation, geometry operations, and file export.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import trimesh
from OCC.Core import gp_Pnt, gp_Vec, gp_Dir, gp_Ax1, gp_Ax2, gp_Trsf
from OCC.Core import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCC.Core import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
from OCC.Core import STEPCADApi_Writer, IGESCADApi_Writer
from OCC.Core import StlAPI_Writer
import structlog

logger = structlog.get_logger()


class CADService:
    """
    Core CAD service for generating 3D models from architectural descriptions.
    Supports complexity levels L0-L3 with progressive feature sets.
    """

    def __init__(self):
        """Initialize the CAD service."""
        self.initialized = False
        self.temp_dir = None
        self.opencascade_loaded = False

    async def initialize(self):
        """Initialize the CAD service and load required libraries."""
        try:
            logger.info("Initializing CAD Service")

            # Create temporary directory for files
            self.temp_dir = Path(tempfile.mkdtemp(prefix="cad_service_"))

            # Test OpenCASCADE availability
            self._test_opencascade()

            self.initialized = True
            logger.info("CAD Service initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize CAD Service", error=str(e))
            raise

    def _test_opencascade(self):
        """Test OpenCASCADE functionality."""
        try:
            # Simple test - create a box
            box = BRepPrimAPI_MakeBox(10, 10, 10).Shape()
            self.opencascade_loaded = True
            logger.info("OpenCASCADE loaded successfully")
        except Exception as e:
            logger.warning("OpenCASCADE not available, falling back to Trimesh", error=str(e))
            self.opencascade_loaded = False

    async def cleanup(self):
        """Cleanup resources and temporary files."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        logger.info("CAD Service cleanup complete")

    async def generate_cad_model(
        self,
        nlp_result: Dict[str, Any],
        complexity_level: str = "L0"
    ) -> Dict[str, Any]:
        """
        Generate a CAD model from NLP processing results.

        Args:
            nlp_result: Processed natural language input with entities and constraints
            complexity_level: L0-L3 complexity level

        Returns:
            Dictionary containing model data, metadata, and export options
        """
        try:
            logger.info("Generating CAD model", complexity=complexity_level)

            if complexity_level == "L0":
                return await self._generate_l0_model(nlp_result)
            elif complexity_level == "L1":
                return await self._generate_l1_model(nlp_result)
            elif complexity_level == "L2":
                return await self._generate_l2_model(nlp_result)
            elif complexity_level == "L3":
                return await self._generate_l3_model(nlp_result)
            else:
                raise ValueError(f"Unsupported complexity level: {complexity_level}")

        except Exception as e:
            logger.error("CAD model generation failed", error=str(e))
            raise

    async def _generate_l0_model(self, nlp_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate L0 - Basic geometric shapes."""
        entities = nlp_result.get("entities", [])

        # Default simple box if no entities
        if not entities:
            mesh = self._create_simple_box(20, 10, 3)
        else:
            # Process first entity for simplicity
            entity = entities[0]
            if entity.get("type") == "building":
                dimensions = entity.get("dimensions", {"length": 20, "width": 10, "height": 3})
                mesh = self._create_simple_box(
                    dimensions.get("length", 20),
                    dimensions.get("width", 10),
                    dimensions.get("height", 3)
                )
            else:
                mesh = self._create_simple_box(20, 10, 3)

        return await self._create_model_response(mesh, nlp_result, "L0")

    async def _generate_l1_model(self, nlp_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate L1 - Residential architecture."""
        # For now, create a simple house structure
        mesh = self._create_simple_house()
        return await self._create_model_response(mesh, nlp_result, "L1")

    async def _generate_l2_model(self, nlp_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate L2 - Commercial buildings."""
        # Create a multi-story building
        mesh = self._create_commercial_building()
        return await self._create_model_response(mesh, nlp_result, "L2")

    async def _generate_l3_model(self, nlp_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate L3 - Iconic structures."""
        # Create a complex parametric structure
        mesh = self._create_parametric_structure()
        return await self._create_model_response(mesh, nlp_result, "L3")

    def _create_simple_box(self, length: float, width: float, height: float) -> trimesh.Trimesh:
        """Create a simple box geometry."""
        box = trimesh.creation.box(extents=[length, width, height])
        return box

    def _create_simple_house(self) -> trimesh.Trimesh:
        """Create a simple house with basic rooms."""
        # Main building box
        main_box = trimesh.creation.box(extents=[15, 10, 3])

        # Roof (pyramid)
        roof = trimesh.creation.cone(radius=8, height=2)
        roof = roof.apply_translation([0, 0, 2.5])

        # Combine
        house = main_box.union(roof)
        return house

    def _create_commercial_building(self) -> trimesh.Trimesh:
        """Create a multi-story commercial building."""
        floors = []
        for i in range(5):  # 5 floors
            floor = trimesh.creation.box(extents=[30, 20, 3])
            floor = floor.apply_translation([0, 0, i * 3])
            floors.append(floor)

        building = floors[0]
        for floor in floors[1:]:
            building = building.union(floor)

        return building

    def _create_parametric_structure(self) -> trimesh.Trimesh:
        """Create a complex parametric structure."""
        # Create a twisted tower
        segments = []
        for i in range(10):
            segment = trimesh.creation.box(extents=[5, 5, 2])
            # Apply rotation and translation
            angle = i * 0.2  # Twist
            transform = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])
            segment = segment.apply_transform(transform)
            segment = segment.apply_translation([0, 0, i * 2])
            segments.append(segment)

        structure = segments[0]
        for segment in segments[1:]:
            structure = structure.union(segment)

        return structure

    async def _create_model_response(
        self,
        mesh: trimesh.Trimesh,
        nlp_result: Dict[str, Any],
        complexity: str
    ) -> Dict[str, Any]:
        """Create standardized model response."""

        # Calculate bounding box
        bounds = mesh.bounds
        bbox = {
            "min": {"x": bounds[0][0], "y": bounds[0][1], "z": bounds[0][2]},
            "max": {"x": bounds[1][0], "y": bounds[1][1], "z": bounds[1][2]}
        }

        # Generate file exports
        export_files = await self._export_model_files(mesh)

        return {
            "id": f"model_{complexity}_{hash(str(nlp_result)) % 10000}",
            "type": "architectural_model",
            "complexity_level": complexity,
            "geometry": {
                "vertices": mesh.vertices.tolist(),
                "faces": mesh.faces.tolist(),
                "normals": mesh.vertex_normals.tolist() if hasattr(mesh, 'vertex_normals') else [],
                "triangle_count": len(mesh.faces),
                "vertex_count": len(mesh.vertices)
            },
            "metadata": {
                "boundingBox": bbox,
                "triangleCount": len(mesh.faces),
                "vertexCount": len(mesh.vertices),
                "volume": float(mesh.volume) if hasattr(mesh, 'volume') else 0,
                "surfaceArea": float(mesh.area) if hasattr(mesh, 'area') else 0,
                "materials": ["concrete"],  # Default material
                "units": "meters"
            },
            "exports": export_files,
            "nlp_context": nlp_result
        }

    async def _export_model_files(self, mesh: trimesh.Trimesh) -> Dict[str, str]:
        """Export model to various file formats."""
        export_files = {}

        try:
            # STL export
            stl_path = self.temp_dir / "model.stl"
            mesh.export(str(stl_path))
            export_files["stl"] = str(stl_path)

            # OBJ export
            obj_path = self.temp_dir / "model.obj"
            mesh.export(str(obj_path))
            export_files["obj"] = str(obj_path)

            # PLY export
            ply_path = self.temp_dir / "model.ply"
            mesh.export(str(ply_path))
            export_files["ply"] = str(ply_path)

            logger.info("Model exported successfully", formats=list(export_files.keys()))

        except Exception as e:
            logger.error("Export failed", error=str(e))

        return export_files

    async def export_to_format(
        self,
        model_data: Dict[str, Any],
        format_type: str
    ) -> str:
        """
        Export model to specific format.

        Args:
            model_data: Model data dictionary
            format_type: Target format (stl, obj, step, iges, ifc, dxf)

        Returns:
            Path to exported file
        """
        try:
            # Reconstruct mesh from model data
            vertices = np.array(model_data["geometry"]["vertices"])
            faces = np.array(model_data["geometry"]["faces"])
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

            # Export based on format
            file_path = self.temp_dir / f"export.{format_type}"

            if format_type.lower() in ["stl", "obj", "ply"]:
                mesh.export(str(file_path))
            else:
                # For advanced formats, create a basic implementation
                # In production, this would use OpenCASCADE
                mesh.export(str(file_path.with_suffix('.stl')))
                logger.warning(f"Format {format_type} exported as STL fallback")
                return str(file_path.with_suffix('.stl'))

            return str(file_path)

        except Exception as e:
            logger.error("Export failed", format=format_type, error=str(e))
            raise

    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats."""
        basic_formats = ["stl", "obj", "ply"]
        advanced_formats = ["step", "iges", "ifc", "dxf"] if self.opencascade_loaded else []
        return basic_formats + advanced_formats

    async def validate_model(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model geometry and metadata."""
        issues = []

        # Check geometry
        vertices = model_data.get("geometry", {}).get("vertices", [])
        faces = model_data.get("geometry", {}).get("faces", [])

        if not vertices:
            issues.append("No vertices found")
        if not faces:
            issues.append("No faces found")

        # Check metadata
        metadata = model_data.get("metadata", {})
        if not metadata.get("boundingBox"):
            issues.append("Missing bounding box")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "model_id": model_data.get("id", "unknown")
        }