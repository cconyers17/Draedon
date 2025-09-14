"""
Data models for architectural NLP processing
Defines core data structures used throughout the pipeline
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime
import numpy as np


class ElementType(Enum):
    """Types of architectural elements"""
    # Structural
    WALL = "wall"
    COLUMN = "column"
    BEAM = "beam"
    SLAB = "slab"
    FOUNDATION = "foundation"
    ROOF = "roof"
    STAIRCASE = "staircase"

    # Spatial
    ROOM = "room"
    FLOOR = "floor"
    ZONE = "zone"
    CORRIDOR = "corridor"

    # Openings
    DOOR = "door"
    WINDOW = "window"
    SKYLIGHT = "skylight"

    # Systems
    HVAC = "hvac"
    ELECTRICAL = "electrical"
    PLUMBING = "plumbing"

    # Furniture
    FURNITURE = "furniture"
    FIXTURE = "fixture"

    # Landscape
    LANDSCAPE = "landscape"

    # Generic
    ELEMENT = "element"

    @classmethod
    def from_string(cls, value: str) -> 'ElementType':
        """Create ElementType from string"""
        value = value.upper()
        if hasattr(cls, value):
            return getattr(cls, value)
        return cls.ELEMENT


class MaterialType(Enum):
    """Types of building materials"""
    CONCRETE = "concrete"
    STEEL = "steel"
    WOOD = "wood"
    GLASS = "glass"
    BRICK = "brick"
    STONE = "stone"
    COMPOSITE = "composite"
    INSULATION = "insulation"
    DRYWALL = "drywall"
    CERAMIC = "ceramic"
    PLASTIC = "plastic"
    ALUMINUM = "aluminum"
    COPPER = "copper"
    OTHER = "other"


class RelationshipType(Enum):
    """Types of spatial relationships"""
    ADJACENT = "adjacent"
    ABOVE = "above"
    BELOW = "below"
    INSIDE = "inside"
    OUTSIDE = "outside"
    CONNECTED = "connected"
    PARALLEL = "parallel"
    PERPENDICULAR = "perpendicular"
    ALIGNED = "aligned"
    OFFSET = "offset"
    INTERSECTING = "intersecting"
    TOUCHING = "touching"
    SEPARATED = "separated"
    FACING = "facing"
    OPPOSITE = "opposite"


class ConstraintType(Enum):
    """Types of design constraints"""
    DIMENSIONAL = "dimensional"
    MATERIAL = "material"
    STRUCTURAL = "structural"
    BUILDING_CODE = "building_code"
    ENVIRONMENTAL = "environmental"
    AESTHETIC = "aesthetic"
    FUNCTIONAL = "functional"
    BUDGET = "budget"
    SCHEDULE = "schedule"
    SAFETY = "safety"
    ACCESSIBILITY = "accessibility"


@dataclass
class Vector3D:
    """3D vector for positions and dimensions"""
    x: float
    y: float
    z: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def magnitude(self) -> float:
        return np.linalg.norm(self.to_array())


@dataclass
class BoundingBox:
    """3D bounding box for elements"""
    min_point: Vector3D
    max_point: Vector3D

    @property
    def center(self) -> Vector3D:
        return Vector3D(
            (self.min_point.x + self.max_point.x) / 2,
            (self.min_point.y + self.max_point.y) / 2,
            (self.min_point.z + self.max_point.z) / 2
        )

    @property
    def dimensions(self) -> Vector3D:
        return Vector3D(
            self.max_point.x - self.min_point.x,
            self.max_point.y - self.min_point.y,
            self.max_point.z - self.min_point.z
        )


@dataclass
class ArchitecturalElement:
    """Represents an architectural element extracted from text"""
    type: ElementType
    name: str
    category: str
    confidence: float
    dimensions: Optional[Dict[str, float]] = None
    material: Optional['MaterialSpecification'] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    position: Optional[Dict[str, Any]] = None
    bounding_box: Optional[BoundingBox] = None
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_basic_geometry(self) -> bool:
        """Check if element is basic geometry"""
        basic_types = [
            ElementType.WALL,
            ElementType.SLAB,
            ElementType.COLUMN,
            ElementType.BEAM
        ]
        return self.type in basic_types

    def get_volume(self) -> Optional[float]:
        """Calculate volume if dimensions available"""
        if self.dimensions and all(k in self.dimensions for k in ['length', 'width', 'height']):
            return self.dimensions['length'] * self.dimensions['width'] * self.dimensions['height']
        return None

    def get_area(self) -> Optional[float]:
        """Calculate area if dimensions available"""
        if self.dimensions:
            if 'area' in self.dimensions:
                return self.dimensions['area']
            elif all(k in self.dimensions for k in ['length', 'width']):
                return self.dimensions['length'] * self.dimensions['width']
        return None


@dataclass
class MaterialSpecification:
    """Specification for building materials"""
    name: str
    type: MaterialType
    properties: Dict[str, Any]
    confidence: float
    density: Optional[float] = None  # kg/m³
    strength: Optional[float] = None  # MPa
    thermal_conductivity: Optional[float] = None  # W/m·K
    cost_per_unit: Optional[float] = None  # $/unit
    fire_rating: Optional[str] = None
    sustainability_rating: Optional[str] = None
    color: Optional[str] = None
    texture: Optional[str] = None
    finish: Optional[str] = None

    def get_weight(self, volume: float) -> Optional[float]:
        """Calculate weight for given volume"""
        if self.density:
            return self.density * volume
        return None


@dataclass
class SpatialRelationship:
    """Represents spatial relationship between elements"""
    source_element: str  # Element ID or name
    target_element: str  # Element ID or name
    relationship_type: RelationshipType
    distance: Optional[float] = None
    angle: Optional[float] = None
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)

    def is_proximity_relation(self) -> bool:
        """Check if this is a proximity relationship"""
        proximity_types = [
            RelationshipType.ADJACENT,
            RelationshipType.TOUCHING,
            RelationshipType.SEPARATED
        ]
        return self.relationship_type in proximity_types

    def is_containment_relation(self) -> bool:
        """Check if this is a containment relationship"""
        containment_types = [
            RelationshipType.INSIDE,
            RelationshipType.OUTSIDE
        ]
        return self.relationship_type in containment_types


@dataclass
class DesignConstraint:
    """Single design constraint"""
    type: ConstraintType
    name: str
    value: Any
    unit: Optional[str] = None
    operator: str = "="  # =, <, >, <=, >=, !=
    priority: int = 1  # 1-5, 5 being highest
    source: Optional[str] = None  # e.g., "IBC 2021", "User requirement"
    description: Optional[str] = None

    def evaluate(self, actual_value: Any) -> bool:
        """Evaluate if constraint is satisfied"""
        if self.operator == "=":
            return actual_value == self.value
        elif self.operator == "<":
            return actual_value < self.value
        elif self.operator == ">":
            return actual_value > self.value
        elif self.operator == "<=":
            return actual_value <= self.value
        elif self.operator == ">=":
            return actual_value >= self.value
        elif self.operator == "!=":
            return actual_value != self.value
        return False


@dataclass
class DesignConstraints:
    """Collection of design constraints"""
    constraints: List[DesignConstraint] = field(default_factory=list)
    building_code: Optional[str] = None
    climate_zone: Optional[str] = None
    seismic_category: Optional[str] = None
    occupancy_type: Optional[str] = None
    max_height: Optional[float] = None
    max_area: Optional[float] = None
    min_clearances: Dict[str, float] = field(default_factory=dict)
    material_restrictions: List[str] = field(default_factory=list)

    def add_constraint(self, constraint: DesignConstraint):
        """Add a constraint to the collection"""
        self.constraints.append(constraint)

    def get_constraints_by_type(self, constraint_type: ConstraintType) -> List[DesignConstraint]:
        """Get all constraints of a specific type"""
        return [c for c in self.constraints if c.type == constraint_type]

    def get_high_priority_constraints(self, min_priority: int = 4) -> List[DesignConstraint]:
        """Get high priority constraints"""
        return [c for c in self.constraints if c.priority >= min_priority]


@dataclass
class ParsedInput:
    """Parsed natural language input"""
    original_text: str
    normalized_text: str
    timestamp: datetime
    language: str = "en"
    tokens: List[str] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DesignContext:
    """Context for design generation"""
    project_type: str  # residential, commercial, industrial
    building_style: Optional[str] = None  # modern, traditional, etc.
    location: Optional[str] = None
    climate: Optional[str] = None
    budget_range: Optional[Tuple[float, float]] = None
    timeline: Optional[str] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    existing_elements: List[ArchitecturalElement] = field(default_factory=list)
    site_constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildingSystem:
    """Represents a building system (HVAC, electrical, etc.)"""
    type: str
    components: List[ArchitecturalElement]
    specifications: Dict[str, Any]
    capacity: Optional[float] = None
    efficiency_rating: Optional[float] = None
    cost_estimate: Optional[float] = None
    maintenance_schedule: Optional[str] = None


@dataclass
class StructuralAnalysis:
    """Results of structural analysis"""
    max_stress: float  # MPa
    max_deflection: float  # mm
    safety_factor: float
    load_distribution: Dict[str, float]
    critical_points: List[Vector3D]
    recommendations: List[str]
    passed: bool


@dataclass
class Room:
    """Represents a room or space"""
    name: str
    type: str  # bedroom, kitchen, etc.
    area: float  # m²
    height: float  # m
    elements: List[ArchitecturalElement]
    adjacencies: List[str]  # Adjacent room IDs
    windows: List[ArchitecturalElement]
    doors: List[ArchitecturalElement]
    furniture: List[ArchitecturalElement]
    occupancy: int  # Number of people
    function: str
    natural_light_score: Optional[float] = None
    ventilation_score: Optional[float] = None


@dataclass
class Floor:
    """Represents a building floor/level"""
    level: int
    name: str
    height: float  # Floor-to-floor height
    area: float  # Total floor area
    rooms: List[Room]
    circulation_area: float
    structural_elements: List[ArchitecturalElement]
    systems: List[BuildingSystem]


@dataclass
class Building:
    """Complete building model"""
    name: str
    type: str
    floors: List[Floor]
    total_area: float
    footprint_area: float
    height: float
    structural_system: str
    envelope: List[ArchitecturalElement]
    systems: List[BuildingSystem]
    site: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_total_volume(self) -> float:
        """Calculate total building volume"""
        return sum(floor.area * floor.height for floor in self.floors)

    def get_room_count(self) -> int:
        """Get total number of rooms"""
        return sum(len(floor.rooms) for floor in self.floors)

    def get_rooms_by_type(self, room_type: str) -> List[Room]:
        """Get all rooms of a specific type"""
        rooms = []
        for floor in self.floors:
            rooms.extend([r for r in floor.rooms if r.type == room_type])
        return rooms