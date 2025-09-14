"""
Database models for CAD projects.
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Text, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.base_class import Base


class Project(Base):
    """
    CAD project model storing project metadata and relationships.
    """
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)

    # Project configuration
    complexity_level = Column(Integer, default=0)  # L0-L3
    project_type = Column(String(50), default="general")  # residential, commercial, industrial
    status = Column(String(50), default="draft")  # draft, processing, completed, failed

    # NLP processing data
    original_text = Column(Text, nullable=True)
    processed_data = Column(JSON, nullable=True)  # Stores NLP extraction results
    intent = Column(String(50), nullable=True)  # CREATE, MODIFY, REMOVE, QUERY

    # CAD model data
    model_data = Column(JSON, nullable=True)  # Stores geometry metadata
    vertices_count = Column(Integer, nullable=True)
    faces_count = Column(Integer, nullable=True)
    volume = Column(Float, nullable=True)  # cubic meters
    surface_area = Column(Float, nullable=True)  # square meters
    bounding_box = Column(JSON, nullable=True)  # min/max coordinates

    # Materials and properties
    materials = Column(JSON, nullable=True)  # List of materials used
    properties = Column(JSON, nullable=True)  # Custom properties

    # Building compliance
    building_code_compliant = Column(Boolean, default=None)
    compliance_issues = Column(JSON, nullable=True)

    # File references
    model_file_path = Column(String(500), nullable=True)
    preview_image_path = Column(String(500), nullable=True)
    export_paths = Column(JSON, nullable=True)  # {format: path} mapping

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Processing metrics
    processing_time_ms = Column(Integer, nullable=True)
    api_calls_count = Column(Integer, default=0)

    # Relationships
    user = relationship("User", back_populates="projects")
    versions = relationship("ProjectVersion", back_populates="project", cascade="all, delete-orphan")
    exports = relationship("ExportHistory", back_populates="project", cascade="all, delete-orphan")


class ProjectVersion(Base):
    """
    Version history for CAD projects.
    """
    __tablename__ = "project_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    version_number = Column(Integer, nullable=False)

    # Version data
    changes = Column(JSON, nullable=False)  # List of modifications
    model_snapshot = Column(JSON, nullable=True)  # Geometry state

    # Metadata
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    comment = Column(Text, nullable=True)

    # Relationships
    project = relationship("Project", back_populates="versions")


class ExportHistory(Base):
    """
    Track export history for projects.
    """
    __tablename__ = "export_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)

    # Export details
    format = Column(String(20), nullable=False)  # STEP, IFC, STL, etc.
    quality = Column(String(20), default="standard")
    units = Column(String(20), default="meters")
    file_size = Column(Integer, nullable=True)  # bytes
    file_path = Column(String(500), nullable=True)

    # User tracking
    exported_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    exported_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    download_count = Column(Integer, default=0)

    # Relationships
    project = relationship("Project", back_populates="exports")


class Material(Base):
    """
    Material database with properties.
    """
    __tablename__ = "materials"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), unique=True, nullable=False, index=True)
    category = Column(String(50), nullable=False)  # concrete, steel, wood, glass, etc.

    # Physical properties
    density = Column(Float, nullable=True)  # kg/m³
    compressive_strength = Column(Float, nullable=True)  # MPa
    tensile_strength = Column(Float, nullable=True)  # MPa
    elastic_modulus = Column(Float, nullable=True)  # GPa
    thermal_conductivity = Column(Float, nullable=True)  # W/m·K

    # Environmental properties
    carbon_footprint = Column(Float, nullable=True)  # kg CO2/kg
    recyclability = Column(Float, nullable=True)  # percentage

    # Cost
    cost_per_unit = Column(Float, nullable=True)  # $/unit
    unit = Column(String(20), default="m³")

    # Additional properties
    properties = Column(JSON, nullable=True)
    fire_rating = Column(String(20), nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class BuildingCode(Base):
    """
    Building codes and regulations database.
    """
    __tablename__ = "building_codes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    code = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    category = Column(String(100), nullable=False)  # structural, fire, accessibility, etc.

    # Code requirements
    requirements = Column(JSON, nullable=False)  # Structured requirements

    # Applicability
    building_types = Column(JSON, nullable=True)  # List of applicable building types
    regions = Column(JSON, nullable=True)  # List of applicable regions

    # Metadata
    version = Column(String(20), nullable=True)
    effective_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)