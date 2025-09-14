"""
Advanced Dimension Processing and Unit Conversion System
Handles complex dimensional expressions and multi-unit conversions
"""

import re
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from fractions import Fraction
import logging

logger = logging.getLogger(__name__)


class DimensionType(Enum):
    """Types of dimensions in architecture"""
    LENGTH = "length"
    WIDTH = "width"
    HEIGHT = "height"
    DEPTH = "depth"
    THICKNESS = "thickness"
    DIAMETER = "diameter"
    RADIUS = "radius"
    AREA = "area"
    VOLUME = "volume"
    ANGLE = "angle"
    SLOPE = "slope"
    SPAN = "span"
    CLEARANCE = "clearance"
    SETBACK = "setback"
    ELEVATION = "elevation"


class UnitSystem(Enum):
    """Measurement unit systems"""
    METRIC = "metric"
    IMPERIAL = "imperial"
    MIXED = "mixed"


@dataclass
class Dimension:
    """Represents a dimensional measurement"""
    value: float
    unit: str
    type: DimensionType
    original_text: str
    confidence: float
    tolerance: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    position: Optional[Tuple[int, int]] = None


@dataclass
class DimensionalExpression:
    """Complex dimensional expression (e.g., 3m x 4m x 2.5m)"""
    dimensions: List[Dimension]
    expression_type: str  # "rectangular", "circular", "irregular"
    original_text: str
    confidence: float


class DimensionProcessor:
    """
    Advanced dimension extraction and processing

    Features:
    - Multi-unit system support (metric, imperial, mixed)
    - Complex expression parsing (e.g., "3m x 4m", "10'-6\"")
    - Fractional dimension handling
    - Range and tolerance extraction
    - Context-aware dimension type inference
    - Automatic unit conversion and normalization
    """

    def __init__(
        self,
        default_unit: str = "meters",
        precision: int = 3,
        unit_system: UnitSystem = UnitSystem.METRIC
    ):
        """Initialize dimension processor"""
        self.default_unit = default_unit
        self.precision = precision
        self.unit_system = unit_system

        # Initialize conversion tables
        self._init_conversion_tables()

        # Compile regex patterns
        self._compile_patterns()

        logger.info(f"Dimension processor initialized with {unit_system.value} system")

    def _init_conversion_tables(self):
        """Initialize unit conversion tables"""
        # Length conversions to meters
        self.to_meters = {
            # Metric
            "mm": 0.001,
            "millimeter": 0.001,
            "millimeters": 0.001,
            "cm": 0.01,
            "centimeter": 0.01,
            "centimeters": 0.01,
            "m": 1.0,
            "meter": 1.0,
            "meters": 1.0,
            "metre": 1.0,
            "metres": 1.0,
            "km": 1000.0,
            "kilometer": 1000.0,
            "kilometers": 1000.0,

            # Imperial
            "in": 0.0254,
            "inch": 0.0254,
            "inches": 0.0254,
            "\"": 0.0254,
            "ft": 0.3048,
            "foot": 0.3048,
            "feet": 0.3048,
            "'": 0.3048,
            "yd": 0.9144,
            "yard": 0.9144,
            "yards": 0.9144,
            "mi": 1609.34,
            "mile": 1609.34,
            "miles": 1609.34
        }

        # Area conversions to square meters
        self.to_sqmeters = {
            "sqm": 1.0,
            "m2": 1.0,
            "m²": 1.0,
            "square meter": 1.0,
            "square meters": 1.0,
            "sqft": 0.092903,
            "ft2": 0.092903,
            "ft²": 0.092903,
            "square foot": 0.092903,
            "square feet": 0.092903,
            "acre": 4046.86,
            "acres": 4046.86,
            "hectare": 10000.0,
            "hectares": 10000.0
        }

        # Volume conversions to cubic meters
        self.to_cubicmeters = {
            "m3": 1.0,
            "m³": 1.0,
            "cubic meter": 1.0,
            "cubic meters": 1.0,
            "ft3": 0.0283168,
            "ft³": 0.0283168,
            "cubic foot": 0.0283168,
            "cubic feet": 0.0283168,
            "liter": 0.001,
            "liters": 0.001,
            "gallon": 0.00378541,
            "gallons": 0.00378541
        }

    def _compile_patterns(self):
        """Compile regex patterns for dimension extraction"""
        # Basic number pattern (including fractions)
        number = r'(?:\d+(?:\.\d+)?|\d+/\d+|\d+\s+\d+/\d+)'

        # Unit pattern
        units = '|'.join(re.escape(unit) for unit in self.to_meters.keys())
        area_units = '|'.join(re.escape(unit) for unit in self.to_sqmeters.keys())
        volume_units = '|'.join(re.escape(unit) for unit in self.to_cubicmeters.keys())

        # Dimension patterns
        self.patterns = {
            # Single dimension: "3m", "10 feet", "5.5 meters"
            'single': re.compile(
                rf'({number})\s*({units})\b',
                re.IGNORECASE
            ),

            # Imperial with feet and inches: "10'-6\"", "10 feet 6 inches"
            'imperial_compound': re.compile(
                r'(\d+)\s*(?:\'|feet|ft)\s*(?:[-\s])?\s*(\d+(?:\.\d+)?)\s*(?:"|inches|in)\b',
                re.IGNORECASE
            ),

            # Rectangular: "3m x 4m", "10ft by 12ft"
            'rectangular': re.compile(
                rf'({number})\s*({units})\s*(?:x|by|×)\s*({number})\s*({units})?',
                re.IGNORECASE
            ),

            # Three dimensional: "3m x 4m x 2.5m"
            'three_dimensional': re.compile(
                rf'({number})\s*({units})\s*(?:x|by|×)\s*({number})\s*({units})?\s*(?:x|by|×)\s*({number})\s*({units})?',
                re.IGNORECASE
            ),

            # Range: "3-5m", "between 10 and 15 feet"
            'range': re.compile(
                rf'(?:({number})\s*[-–]\s*({number})|between\s+({number})\s+and\s+({number}))\s*({units})',
                re.IGNORECASE
            ),

            # Tolerance: "3m ± 0.1m", "10ft +/- 0.5ft"
            'tolerance': re.compile(
                rf'({number})\s*({units})\s*(?:±|\+/-)\s*({number})\s*({units})?',
                re.IGNORECASE
            ),

            # Area: "100 sqm", "500 square feet"
            'area': re.compile(
                rf'({number})\s*({area_units})',
                re.IGNORECASE
            ),

            # Volume: "50 m3", "1000 cubic feet"
            'volume': re.compile(
                rf'({number})\s*({volume_units})',
                re.IGNORECASE
            ),

            # Diameter/Radius: "diameter of 5m", "10ft diameter", "radius 2.5m"
            'circular': re.compile(
                rf'(?:diameter|radius|dia|rad)\s*(?:of\s*)?({number})\s*({units})|({number})\s*({units})\s*(?:diameter|radius|dia|rad)',
                re.IGNORECASE
            ),

            # Approximate: "about 3m", "approximately 10 feet", "roughly 5m"
            'approximate': re.compile(
                rf'(?:about|approximately|roughly|around|circa|~)\s*({number})\s*({units})',
                re.IGNORECASE
            )
        }

    async def extract_dimensions(self, text: str) -> Dict[str, Any]:
        """
        Extract all dimensions from text

        Args:
            text: Input text containing dimensions

        Returns:
            Dictionary of extracted dimensions with metadata
        """
        dimensions = {
            "linear": [],
            "area": [],
            "volume": [],
            "ranges": [],
            "tolerances": [],
            "expressions": [],
            "all": []
        }

        # Normalize text
        text = self._normalize_text(text)

        # Extract different types of dimensions
        linear_dims = self._extract_linear_dimensions(text)
        dimensions["linear"].extend(linear_dims)

        area_dims = self._extract_area_dimensions(text)
        dimensions["area"].extend(area_dims)

        volume_dims = self._extract_volume_dimensions(text)
        dimensions["volume"].extend(volume_dims)

        range_dims = self._extract_range_dimensions(text)
        dimensions["ranges"].extend(range_dims)

        tolerance_dims = self._extract_tolerance_dimensions(text)
        dimensions["tolerances"].extend(tolerance_dims)

        # Extract complex expressions
        expressions = self._extract_dimensional_expressions(text)
        dimensions["expressions"].extend(expressions)

        # Combine all dimensions
        dimensions["all"] = (
            dimensions["linear"] +
            dimensions["area"] +
            dimensions["volume"]
        )

        # Infer dimension types from context
        dimensions = self._infer_dimension_types(dimensions, text)

        # Convert to standard units
        dimensions = self._standardize_units(dimensions)

        # Add metadata
        dimensions["metadata"] = {
            "unit_system": self.unit_system.value,
            "default_unit": self.default_unit,
            "total_count": len(dimensions["all"])
        }

        return dimensions

    def _normalize_text(self, text: str) -> str:
        """Normalize text for dimension extraction"""
        # Replace unicode symbols
        replacements = {
            "×": "x",
            "–": "-",
            "'": "'",
            """: "\"",
            """: "\"",
            "²": "2",
            "³": "3"
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def _extract_linear_dimensions(self, text: str) -> List[Dimension]:
        """Extract linear dimensions from text"""
        dimensions = []

        # Check for imperial compound (feet and inches)
        for match in self.patterns['imperial_compound'].finditer(text):
            feet = float(match.group(1))
            inches = float(match.group(2))
            total_meters = feet * 0.3048 + inches * 0.0254

            dimensions.append(Dimension(
                value=total_meters,
                unit="meters",
                type=DimensionType.LENGTH,
                original_text=match.group(0),
                confidence=0.95,
                position=(match.start(), match.end())
            ))

        # Single dimensions
        for match in self.patterns['single'].finditer(text):
            value_str = match.group(1)
            unit = match.group(2).lower()

            # Parse value (handle fractions)
            value = self._parse_number(value_str)

            # Convert to meters
            if unit in self.to_meters:
                value_meters = value * self.to_meters[unit]

                dimension = Dimension(
                    value=value_meters,
                    unit="meters",
                    type=DimensionType.LENGTH,
                    original_text=match.group(0),
                    confidence=0.9,
                    position=(match.start(), match.end())
                )

                dimensions.append(dimension)

        # Circular dimensions (diameter/radius)
        for match in self.patterns['circular'].finditer(text):
            if match.group(1) and match.group(2):
                value = self._parse_number(match.group(1))
                unit = match.group(2).lower()
            else:
                value = self._parse_number(match.group(3))
                unit = match.group(4).lower()

            if unit in self.to_meters:
                value_meters = value * self.to_meters[unit]

                # Determine if diameter or radius
                if "diameter" in match.group(0).lower() or "dia" in match.group(0).lower():
                    dim_type = DimensionType.DIAMETER
                else:
                    dim_type = DimensionType.RADIUS

                dimensions.append(Dimension(
                    value=value_meters,
                    unit="meters",
                    type=dim_type,
                    original_text=match.group(0),
                    confidence=0.9,
                    position=(match.start(), match.end())
                ))

        return dimensions

    def _extract_area_dimensions(self, text: str) -> List[Dimension]:
        """Extract area dimensions from text"""
        dimensions = []

        for match in self.patterns['area'].finditer(text):
            value = self._parse_number(match.group(1))
            unit = match.group(2).lower()

            if unit in self.to_sqmeters:
                value_sqm = value * self.to_sqmeters[unit]

                dimensions.append(Dimension(
                    value=value_sqm,
                    unit="square_meters",
                    type=DimensionType.AREA,
                    original_text=match.group(0),
                    confidence=0.9,
                    position=(match.start(), match.end())
                ))

        return dimensions

    def _extract_volume_dimensions(self, text: str) -> List[Dimension]:
        """Extract volume dimensions from text"""
        dimensions = []

        for match in self.patterns['volume'].finditer(text):
            value = self._parse_number(match.group(1))
            unit = match.group(2).lower()

            if unit in self.to_cubicmeters:
                value_m3 = value * self.to_cubicmeters[unit]

                dimensions.append(Dimension(
                    value=value_m3,
                    unit="cubic_meters",
                    type=DimensionType.VOLUME,
                    original_text=match.group(0),
                    confidence=0.9,
                    position=(match.start(), match.end())
                ))

        return dimensions

    def _extract_range_dimensions(self, text: str) -> List[Dimension]:
        """Extract dimension ranges from text"""
        dimensions = []

        for match in self.patterns['range'].finditer(text):
            if match.group(1) and match.group(2):
                # Format: "3-5m"
                min_val = self._parse_number(match.group(1))
                max_val = self._parse_number(match.group(2))
                unit = match.group(5).lower()
            else:
                # Format: "between 3 and 5m"
                min_val = self._parse_number(match.group(3))
                max_val = self._parse_number(match.group(4))
                unit = match.group(5).lower()

            if unit in self.to_meters:
                min_meters = min_val * self.to_meters[unit]
                max_meters = max_val * self.to_meters[unit]
                avg_meters = (min_meters + max_meters) / 2

                dimensions.append(Dimension(
                    value=avg_meters,
                    unit="meters",
                    type=DimensionType.LENGTH,
                    original_text=match.group(0),
                    confidence=0.85,
                    min_value=min_meters,
                    max_value=max_meters,
                    position=(match.start(), match.end())
                ))

        return dimensions

    def _extract_tolerance_dimensions(self, text: str) -> List[Dimension]:
        """Extract dimensions with tolerances"""
        dimensions = []

        for match in self.patterns['tolerance'].finditer(text):
            value = self._parse_number(match.group(1))
            unit = match.group(2).lower()
            tolerance = self._parse_number(match.group(3))
            tol_unit = match.group(4).lower() if match.group(4) else unit

            if unit in self.to_meters:
                value_meters = value * self.to_meters[unit]
                tolerance_meters = tolerance * self.to_meters[tol_unit]

                dimensions.append(Dimension(
                    value=value_meters,
                    unit="meters",
                    type=DimensionType.LENGTH,
                    original_text=match.group(0),
                    confidence=0.9,
                    tolerance=tolerance_meters,
                    min_value=value_meters - tolerance_meters,
                    max_value=value_meters + tolerance_meters,
                    position=(match.start(), match.end())
                ))

        return dimensions

    def _extract_dimensional_expressions(self, text: str) -> List[DimensionalExpression]:
        """Extract complex dimensional expressions"""
        expressions = []

        # Three dimensional expressions
        for match in self.patterns['three_dimensional'].finditer(text):
            dims = []

            # Extract each dimension
            for i in range(3):
                value = self._parse_number(match.group(i * 2 + 1))
                unit = match.group(i * 2 + 2)
                if not unit and i > 0:
                    unit = match.group(2)  # Use first unit if not specified

                if unit:
                    unit = unit.lower()
                    if unit in self.to_meters:
                        value_meters = value * self.to_meters[unit]
                        dims.append(Dimension(
                            value=value_meters,
                            unit="meters",
                            type=[DimensionType.LENGTH, DimensionType.WIDTH, DimensionType.HEIGHT][i],
                            original_text=f"{value}{unit}",
                            confidence=0.9
                        ))

            if len(dims) == 3:
                expressions.append(DimensionalExpression(
                    dimensions=dims,
                    expression_type="rectangular",
                    original_text=match.group(0),
                    confidence=0.9
                ))

        # Two dimensional expressions
        for match in self.patterns['rectangular'].finditer(text):
            # Skip if already matched as 3D
            if any(match.group(0) in expr.original_text for expr in expressions):
                continue

            dims = []

            value1 = self._parse_number(match.group(1))
            unit1 = match.group(2).lower()
            value2 = self._parse_number(match.group(3))
            unit2 = match.group(4).lower() if match.group(4) else unit1

            if unit1 in self.to_meters:
                value1_meters = value1 * self.to_meters[unit1]
                dims.append(Dimension(
                    value=value1_meters,
                    unit="meters",
                    type=DimensionType.LENGTH,
                    original_text=f"{value1}{unit1}",
                    confidence=0.9
                ))

            if unit2 in self.to_meters:
                value2_meters = value2 * self.to_meters[unit2]
                dims.append(Dimension(
                    value=value2_meters,
                    unit="meters",
                    type=DimensionType.WIDTH,
                    original_text=f"{value2}{unit2}",
                    confidence=0.9
                ))

            if len(dims) == 2:
                expressions.append(DimensionalExpression(
                    dimensions=dims,
                    expression_type="rectangular",
                    original_text=match.group(0),
                    confidence=0.9
                ))

        return expressions

    def _parse_number(self, value_str: str) -> float:
        """Parse number string including fractions"""
        value_str = value_str.strip()

        # Handle mixed numbers (e.g., "1 1/2")
        if ' ' in value_str and '/' in value_str:
            parts = value_str.split(' ')
            whole = float(parts[0])
            fraction = Fraction(parts[1])
            return whole + float(fraction)

        # Handle fractions
        if '/' in value_str:
            return float(Fraction(value_str))

        # Regular number
        return float(value_str)

    def _infer_dimension_types(
        self, dimensions: Dict[str, Any], text: str
    ) -> Dict[str, Any]:
        """Infer dimension types from context"""
        text_lower = text.lower()

        # Context keywords for dimension types
        type_keywords = {
            DimensionType.HEIGHT: ["height", "tall", "high", "elevation", "rise"],
            DimensionType.WIDTH: ["width", "wide", "breadth", "span"],
            DimensionType.LENGTH: ["length", "long", "deep"],
            DimensionType.DEPTH: ["depth", "deep", "below", "foundation"],
            DimensionType.THICKNESS: ["thickness", "thick", "thin"],
            DimensionType.CLEARANCE: ["clearance", "headroom", "clear height"],
            DimensionType.SETBACK: ["setback", "offset", "distance from"],
            DimensionType.SPAN: ["span", "spanning", "beam span", "joist span"],
            DimensionType.SLOPE: ["slope", "pitch", "grade", "incline", "gradient"]
        }

        # Update dimension types based on context
        for dim_list in [dimensions["linear"], dimensions["all"]]:
            for dim in dim_list:
                if dim.type == DimensionType.LENGTH:  # Default type
                    # Check context for specific type
                    for dim_type, keywords in type_keywords.items():
                        for keyword in keywords:
                            # Look for keyword near dimension
                            if dim.position:
                                start = max(0, dim.position[0] - 50)
                                end = min(len(text), dim.position[1] + 50)
                                context = text_lower[start:end]

                                if keyword in context:
                                    dim.type = dim_type
                                    break

        return dimensions

    def _standardize_units(self, dimensions: Dict[str, Any]) -> Dict[str, Any]:
        """Convert all dimensions to standard units based on system"""
        if self.unit_system == UnitSystem.IMPERIAL:
            # Convert meters to feet
            conversion_factor = 3.28084
            target_unit = "feet"
        else:
            # Keep in meters
            conversion_factor = 1.0
            target_unit = "meters"

        # Convert linear dimensions
        for dim in dimensions["linear"]:
            if dim.unit == "meters" and self.unit_system == UnitSystem.IMPERIAL:
                dim.value = round(dim.value * conversion_factor, self.precision)
                dim.unit = target_unit

                if dim.min_value:
                    dim.min_value = round(dim.min_value * conversion_factor, self.precision)
                if dim.max_value:
                    dim.max_value = round(dim.max_value * conversion_factor, self.precision)
                if dim.tolerance:
                    dim.tolerance = round(dim.tolerance * conversion_factor, self.precision)

        # Convert area dimensions
        if self.unit_system == UnitSystem.IMPERIAL:
            area_conversion = 10.7639  # sqm to sqft
            area_unit = "square_feet"
        else:
            area_conversion = 1.0
            area_unit = "square_meters"

        for dim in dimensions["area"]:
            if dim.unit == "square_meters" and self.unit_system == UnitSystem.IMPERIAL:
                dim.value = round(dim.value * area_conversion, self.precision)
                dim.unit = area_unit

        # Convert volume dimensions
        if self.unit_system == UnitSystem.IMPERIAL:
            volume_conversion = 35.3147  # m3 to ft3
            volume_unit = "cubic_feet"
        else:
            volume_conversion = 1.0
            volume_unit = "cubic_meters"

        for dim in dimensions["volume"]:
            if dim.unit == "cubic_meters" and self.unit_system == UnitSystem.IMPERIAL:
                dim.value = round(dim.value * volume_conversion, self.precision)
                dim.unit = volume_unit

        return dimensions

    def convert_dimension(
        self,
        value: float,
        from_unit: str,
        to_unit: str
    ) -> float:
        """Convert dimension between units"""
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()

        # Check if units are in conversion tables
        if from_unit in self.to_meters and to_unit in self.to_meters:
            # Convert through meters
            meters = value * self.to_meters[from_unit]
            result = meters / self.to_meters[to_unit]
            return round(result, self.precision)

        elif from_unit in self.to_sqmeters and to_unit in self.to_sqmeters:
            # Convert through square meters
            sqm = value * self.to_sqmeters[from_unit]
            result = sqm / self.to_sqmeters[to_unit]
            return round(result, self.precision)

        elif from_unit in self.to_cubicmeters and to_unit in self.to_cubicmeters:
            # Convert through cubic meters
            m3 = value * self.to_cubicmeters[from_unit]
            result = m3 / self.to_cubicmeters[to_unit]
            return round(result, self.precision)

        else:
            raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")

    def parse_dimension_string(self, dim_str: str) -> Optional[Dimension]:
        """Parse a single dimension string"""
        # Try each pattern
        for pattern_name, pattern in self.patterns.items():
            match = pattern.match(dim_str.strip())
            if match:
                # Process based on pattern type
                if pattern_name == 'single':
                    value = self._parse_number(match.group(1))
                    unit = match.group(2).lower()

                    if unit in self.to_meters:
                        value_meters = value * self.to_meters[unit]
                        return Dimension(
                            value=value_meters,
                            unit="meters",
                            type=DimensionType.LENGTH,
                            original_text=dim_str,
                            confidence=0.95
                        )

        return None

    def format_dimension(
        self,
        dimension: Dimension,
        unit_system: Optional[UnitSystem] = None
    ) -> str:
        """Format dimension for display"""
        if not unit_system:
            unit_system = self.unit_system

        if unit_system == UnitSystem.IMPERIAL:
            # Convert to feet and inches
            feet = dimension.value * 3.28084
            whole_feet = int(feet)
            inches = (feet - whole_feet) * 12

            if inches < 0.5:
                return f"{whole_feet}'"
            else:
                return f"{whole_feet}'-{inches:.0f}\""
        else:
            # Format in meters
            if dimension.value < 0.01:
                return f"{dimension.value * 1000:.0f}mm"
            elif dimension.value < 1:
                return f"{dimension.value * 100:.0f}cm"
            else:
                return f"{dimension.value:.2f}m"