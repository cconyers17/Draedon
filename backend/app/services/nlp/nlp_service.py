"""
NLP Processing Service for architectural text understanding.
"""

# import spacy  # Commented out for deployment
from typing import Dict, List, Any, Optional
import re
from datetime import datetime
import structlog
# from transformers import pipeline  # Commented out for deployment
import nltk

from app.core.config import settings
from app.utils.cache import CacheDecorator

logger = structlog.get_logger()


class NLPService:
    """
    Service for processing natural language architectural descriptions.
    """

    def __init__(self):
        self.nlp = None
        self.classifier = None
        self.entity_patterns = self._load_entity_patterns()
        self.dimension_patterns = self._compile_dimension_patterns()
        self.building_codes = self._load_building_codes()

    async def initialize(self):
        """
        Initialize NLP models and resources.
        """
        try:
            # Simplified NLP for deployment - using NLTK instead of spaCy
            # self.nlp = spacy.load(settings.SPACY_MODEL)

            # Download required NLTK data
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
            except:
                pass  # Ignore download errors in production

            self.nlp = None  # Will use simple regex-based processing
            self.classifier = None  # Will use simple keyword classification

            logger.info("NLP service initialized successfully (simplified mode)")

        except Exception as e:
            logger.error("Failed to initialize NLP service", error=str(e))
            raise

    async def cleanup(self):
        """
        Cleanup NLP resources.
        """
        self.nlp = None
        self.classifier = None

    def _load_entity_patterns(self) -> List[Dict]:
        """
        Load architectural entity patterns for recognition.
        """
        return [
            # Building elements
            {"label": "BUILDING_ELEMENT", "pattern": "wall"},
            {"label": "BUILDING_ELEMENT", "pattern": "door"},
            {"label": "BUILDING_ELEMENT", "pattern": "window"},
            {"label": "BUILDING_ELEMENT", "pattern": "roof"},
            {"label": "BUILDING_ELEMENT", "pattern": "floor"},
            {"label": "BUILDING_ELEMENT", "pattern": "ceiling"},
            {"label": "BUILDING_ELEMENT", "pattern": "column"},
            {"label": "BUILDING_ELEMENT", "pattern": "beam"},
            {"label": "BUILDING_ELEMENT", "pattern": "stair"},
            {"label": "BUILDING_ELEMENT", "pattern": "elevator"},

            # Spaces
            {"label": "SPACE", "pattern": "room"},
            {"label": "SPACE", "pattern": "office"},
            {"label": "SPACE", "pattern": "bathroom"},
            {"label": "SPACE", "pattern": "kitchen"},
            {"label": "SPACE", "pattern": "bedroom"},
            {"label": "SPACE", "pattern": "living room"},
            {"label": "SPACE", "pattern": "hallway"},
            {"label": "SPACE", "pattern": "lobby"},
            {"label": "SPACE", "pattern": "conference room"},

            # Materials
            {"label": "MATERIAL", "pattern": "concrete"},
            {"label": "MATERIAL", "pattern": "steel"},
            {"label": "MATERIAL", "pattern": "wood"},
            {"label": "MATERIAL", "pattern": "glass"},
            {"label": "MATERIAL", "pattern": "brick"},
            {"label": "MATERIAL", "pattern": "stone"},
            {"label": "MATERIAL", "pattern": "aluminum"},
            {"label": "MATERIAL", "pattern": "drywall"},

            # Locations
            {"label": "LOCATION", "pattern": "north"},
            {"label": "LOCATION", "pattern": "south"},
            {"label": "LOCATION", "pattern": "east"},
            {"label": "LOCATION", "pattern": "west"},
            {"label": "LOCATION", "pattern": "center"},
            {"label": "LOCATION", "pattern": "corner"},
            {"label": "LOCATION", "pattern": "front"},
            {"label": "LOCATION", "pattern": "back"},
            {"label": "LOCATION", "pattern": "side"},
        ]

    def _compile_dimension_patterns(self) -> List[re.Pattern]:
        """
        Compile regex patterns for dimension extraction.
        """
        return [
            re.compile(r'(\d+(?:\.\d+)?)\s*(m|meter|meters|metre|metres)', re.IGNORECASE),
            re.compile(r'(\d+(?:\.\d+)?)\s*(ft|foot|feet)', re.IGNORECASE),
            re.compile(r'(\d+(?:\.\d+)?)\s*(in|inch|inches)', re.IGNORECASE),
            re.compile(r'(\d+(?:\.\d+)?)\s*(cm|centimeter|centimeters)', re.IGNORECASE),
            re.compile(r'(\d+(?:\.\d+)?)\s*(mm|millimeter|millimeters)', re.IGNORECASE),
            re.compile(r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)', re.IGNORECASE),  # 10x15 format
        ]

    def _load_building_codes(self) -> Dict[str, Any]:
        """
        Load building code constraints.
        """
        return {
            "MIN_CEILING_HEIGHT": 2.4,  # meters
            "MIN_ROOM_AREA": 7.0,  # square meters
            "MIN_DOOR_WIDTH": 0.8,  # meters
            "MAX_DOOR_WIDTH": 1.2,  # meters
            "MIN_WINDOW_AREA_RATIO": 0.1,  # 10% of floor area
            "MIN_CORRIDOR_WIDTH": 1.2,  # meters
            "MIN_STAIR_WIDTH": 0.9,  # meters
            "MAX_FLOOR_HEIGHT": 4.5,  # meters
        }

    @CacheDecorator(ttl=settings.NLP_CACHE_TTL, key_prefix="nlp:process")
    async def process_text(
        self,
        text: str,
        complexity_level: int = 0,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process architectural text to extract structured data.
        """
        start_time = datetime.utcnow()

        try:
            # Simplified processing without spaCy
            # doc = self.nlp(text)

            # Extract entities using simple keyword matching
            entities = await self.extract_entities(text)

            # Extract dimensions
            dimensions = self._extract_dimensions(text)

            # Classify intent using simple keywords
            intent = await self.classify_intent(text)

            # Extract constraints using basic patterns
            constraints = self._extract_constraints_simple(text, entities)

            # Validate against building codes
            validation = await self.validate_constraints(entities, context.get("building_type", "general") if context else "general")

            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            return {
                "success": True,
                "intent": intent["intent"],
                "entities": entities,
                "dimensions": dimensions,
                "constraints": constraints,
                "validation": validation,
                "complexity_level": complexity_level,
                "processing_time_ms": int(processing_time)
            }

        except Exception as e:
            logger.error("Text processing failed", error=str(e), text=text[:100])
            raise

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract architectural entities from text using simple keyword matching.
        """
        entities = []
        text_lower = text.lower()

        # Simple entity extraction using keyword patterns
        entity_keywords = {
            "BUILDING_ELEMENT": ["wall", "door", "window", "roof", "floor", "column", "beam"],
            "SPACE": ["room", "office", "bathroom", "kitchen", "bedroom", "living room", "hallway"],
            "MATERIAL": ["concrete", "steel", "wood", "glass", "brick", "stone"],
            "LOCATION": ["north", "south", "east", "west", "center", "corner"]
        }

        for entity_type, keywords in entity_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    start_pos = text_lower.find(keyword)
                    entity = {
                        "type": entity_type,
                        "value": keyword,
                        "start": start_pos,
                        "end": start_pos + len(keyword),
                        "confidence": 0.9
                    }

                    # Extract associated dimensions if present
                    dimensions = self._extract_entity_dimensions(text[max(0, start_pos - 20):min(len(text), start_pos + len(keyword) + 20)])
                    if dimensions:
                        entity["dimensions"] = dimensions

                    # Extract quantity
                    quantity = self._extract_quantity(text[max(0, start_pos - 10):start_pos])
                    if quantity:
                        entity["quantity"] = quantity

                    entities.append(entity)

        return entities

    async def classify_intent(self, text: str) -> Dict[str, Any]:
        """
        Classify user intent from text using simple keyword matching.
        """
        text_lower = text.lower()

        # Simple intent classification using keywords
        if any(word in text_lower for word in ["create", "build", "design", "make", "generate"]):
            intent = "CREATE"
            confidence = 0.9
        elif any(word in text_lower for word in ["modify", "change", "update", "edit", "alter"]):
            intent = "MODIFY"
            confidence = 0.8
        elif any(word in text_lower for word in ["remove", "delete", "eliminate"]):
            intent = "REMOVE"
            confidence = 0.8
        else:
            intent = "CREATE"  # Default to create for architectural descriptions
            confidence = 0.7

        # Determine sub-intent based on keywords
        sub_intent = self._determine_sub_intent(text, intent)

        return {
            "intent": intent,
            "confidence": confidence,
            "sub_intent": sub_intent
        }

    def _determine_sub_intent(self, text: str, intent: str) -> Optional[str]:
        """
        Determine specific sub-intent based on text content.
        """
        text_lower = text.lower()

        if intent == "CREATE":
            if "building" in text_lower:
                return "create_building"
            elif "room" in text_lower or "space" in text_lower:
                return "create_space"
            elif any(elem in text_lower for elem in ["wall", "door", "window"]):
                return "add_element"

        elif intent == "MODIFY":
            if "resize" in text_lower or "change size" in text_lower:
                return "resize"
            elif "move" in text_lower or "relocate" in text_lower:
                return "relocate"
            elif "material" in text_lower:
                return "change_material"

        elif intent == "REMOVE":
            if "all" in text_lower:
                return "remove_all"
            else:
                return "remove_element"

        return None

    def _extract_dimensions(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract dimensions from text.
        """
        dimensions = []

        for pattern in self.dimension_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if len(match) == 2:
                    value, unit = match
                    dimensions.append({
                        "value": float(value),
                        "unit": self._normalize_unit(unit),
                        "original": f"{value} {unit}"
                    })

        return dimensions

    def _normalize_unit(self, unit: str) -> str:
        """
        Normalize measurement units.
        """
        unit_lower = unit.lower()
        if unit_lower in ["m", "meter", "meters", "metre", "metres"]:
            return "meters"
        elif unit_lower in ["ft", "foot", "feet"]:
            return "feet"
        elif unit_lower in ["in", "inch", "inches"]:
            return "inches"
        elif unit_lower in ["cm", "centimeter", "centimeters"]:
            return "centimeters"
        elif unit_lower in ["mm", "millimeter", "millimeters"]:
            return "millimeters"
        return unit

    def _extract_entity_dimensions(self, context: str) -> Optional[Dict[str, Any]]:
        """
        Extract dimensions associated with an entity.
        """
        dimensions = self._extract_dimensions(context)
        if dimensions:
            # Convert to standard format
            if len(dimensions) >= 2:
                return {
                    "length": dimensions[0]["value"],
                    "width": dimensions[1]["value"],
                    "unit": dimensions[0]["unit"]
                }
            elif len(dimensions) == 1:
                return {
                    "value": dimensions[0]["value"],
                    "unit": dimensions[0]["unit"]
                }
        return None

    def _extract_quantity(self, context: str) -> Optional[int]:
        """
        Extract quantity from context.
        """
        quantity_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }

        context_lower = context.lower()
        for word, value in quantity_words.items():
            if word in context_lower:
                return value

        # Check for numeric values
        numbers = re.findall(r'\d+', context)
        if numbers:
            return int(numbers[-1])

        return None

    def _extract_constraints(self, doc, entities: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extract spatial and structural constraints.
        """
        constraints = []

        # Extract spatial relationships
        for token in doc:
            if token.dep_ == "prep" and token.text.lower() in ["on", "in", "at", "near", "beside", "between"]:
                constraint = {
                    "type": "spatial",
                    "relationship": token.text.lower(),
                    "description": f"{token.head.text} {token.text} {' '.join([child.text for child in token.children])}"
                }
                constraints.append(constraint)

        return constraints

    def _extract_constraints_simple(self, text: str, entities: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extract spatial and structural constraints using simple patterns.
        """
        constraints = []
        text_lower = text.lower()

        # Simple spatial relationship patterns
        spatial_patterns = {
            "adjacent": ["next to", "beside", "adjacent to"],
            "inside": ["in", "inside", "within"],
            "on": ["on", "above", "on top of"],
            "near": ["near", "close to", "around"]
        }

        for relationship, keywords in spatial_patterns.items():
            for keyword in keywords:
                if keyword in text_lower:
                    constraint = {
                        "type": "spatial",
                        "relationship": relationship,
                        "description": f"Elements have {relationship} relationship"
                    }
                    constraints.append(constraint)

        return constraints

    async def validate_constraints(
        self,
        entities: List[Dict[str, Any]],
        building_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Validate entities against building codes.
        """
        warnings = []
        errors = []

        for entity in entities:
            if entity["type"] == "SPACE" and "dimensions" in entity:
                dims = entity["dimensions"]
                if "length" in dims and "width" in dims:
                    area = dims["length"] * dims["width"]
                    if area < self.building_codes["MIN_ROOM_AREA"]:
                        warnings.append(f"Room area {area}m² is below minimum requirement of {self.building_codes['MIN_ROOM_AREA']}m²")

            elif entity["type"] == "BUILDING_ELEMENT":
                if entity["value"] == "door" and "dimensions" in entity:
                    width = entity["dimensions"].get("width", entity["dimensions"].get("value"))
                    if width:
                        if width < self.building_codes["MIN_DOOR_WIDTH"]:
                            errors.append(f"Door width {width}m is below minimum {self.building_codes['MIN_DOOR_WIDTH']}m")
                        elif width > self.building_codes["MAX_DOOR_WIDTH"]:
                            warnings.append(f"Door width {width}m exceeds standard maximum {self.building_codes['MAX_DOOR_WIDTH']}m")

        return {
            "building_code_compliant": len(errors) == 0,
            "warnings": warnings,
            "errors": errors
        }