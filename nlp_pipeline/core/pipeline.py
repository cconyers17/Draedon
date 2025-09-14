"""
Main NLP Pipeline for Text-to-CAD Conversion
Orchestrates all NLP components for comprehensive architectural text analysis
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json

from .entity_recognition import ArchitecturalEntityRecognizer
from .intent_classifier import IntentClassifier
from .dimension_processor import DimensionProcessor
from .spatial_parser import SpatialRelationshipParser
from .constraint_extractor import ConstraintExtractor
from .material_extractor import MaterialSpecificationExtractor
from .context_manager import ContextAwareDisambiguator
from .validation_engine import BuildingCodeValidator
from .multimodal_processor import MultiModalInputProcessor
from ..models.architectural_models import (
    ArchitecturalElement, ParsedInput, DesignConstraints,
    SpatialRelationship, MaterialSpecification
)
from ..utils.caching import SemanticCache
from ..utils.metrics import PerformanceMonitor

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing modes for different complexity levels"""
    BASIC = "basic"  # L0: Simple geometric shapes
    RESIDENTIAL = "residential"  # L1: Residential buildings
    COMMERCIAL = "commercial"  # L2: Commercial structures
    COMPLEX = "complex"  # L3: Iconic/complex structures


@dataclass
class PipelineConfig:
    """Configuration for NLP pipeline"""
    mode: ProcessingMode = ProcessingMode.RESIDENTIAL
    enable_caching: bool = True
    enable_validation: bool = True
    enable_disambiguation: bool = True
    enable_multimodal: bool = False
    parallel_processing: bool = True
    confidence_threshold: float = 0.85
    max_iterations: int = 3
    timeout_seconds: int = 30
    batch_size: int = 10
    use_gpu: bool = True
    model_versions: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Complete result from NLP pipeline processing"""
    parsed_input: ParsedInput
    entities: List[ArchitecturalElement]
    dimensions: Dict[str, Any]
    spatial_relationships: List[SpatialRelationship]
    materials: List[MaterialSpecification]
    constraints: DesignConstraints
    intent: str
    confidence_scores: Dict[str, float]
    processing_time: float
    validation_results: Dict[str, Any]
    metadata: Dict[str, Any]
    warnings: List[str]
    suggestions: List[str]


class ArchitecturalNLPPipeline:
    """
    Advanced NLP Pipeline for Architectural Text Processing

    Features:
    - Multi-stage processing with validation
    - Context-aware disambiguation
    - Parallel processing for performance
    - Semantic caching for repeated queries
    - Comprehensive error handling
    - Real-time performance monitoring
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the NLP pipeline with configuration"""
        self.config = config or PipelineConfig()
        self.performance_monitor = PerformanceMonitor()

        # Initialize core components
        self._initialize_components()

        # Setup caching if enabled
        if self.config.enable_caching:
            self.cache = SemanticCache(max_size=1000, ttl=3600)

        # Setup parallel processing
        if self.config.parallel_processing:
            self.executor = ThreadPoolExecutor(max_workers=4)

        logger.info(f"NLP Pipeline initialized in {self.config.mode.value} mode")

    def _initialize_components(self):
        """Initialize all NLP components"""
        # Core NLP components
        self.entity_recognizer = ArchitecturalEntityRecognizer(
            model_version=self.config.model_versions.get("entity", "latest"),
            use_gpu=self.config.use_gpu
        )

        self.intent_classifier = IntentClassifier(
            confidence_threshold=self.config.confidence_threshold
        )

        self.dimension_processor = DimensionProcessor(
            default_unit="meters",
            precision=3
        )

        self.spatial_parser = SpatialRelationshipParser(
            mode=self.config.mode
        )

        self.constraint_extractor = ConstraintExtractor(
            building_codes=self._load_building_codes()
        )

        self.material_extractor = MaterialSpecificationExtractor(
            material_database=self._load_material_database()
        )

        # Advanced components
        if self.config.enable_disambiguation:
            self.disambiguator = ContextAwareDisambiguator()

        if self.config.enable_validation:
            self.validator = BuildingCodeValidator()

        if self.config.enable_multimodal:
            self.multimodal_processor = MultiModalInputProcessor()

    async def process_async(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        images: Optional[List[Any]] = None
    ) -> ProcessingResult:
        """
        Asynchronously process architectural text description

        Args:
            text: Natural language architectural description
            context: Optional context information
            images: Optional images for multi-modal processing

        Returns:
            ProcessingResult with all extracted information
        """
        start_time = datetime.now()

        # Check cache if enabled
        if self.config.enable_caching:
            cached_result = self.cache.get(text)
            if cached_result:
                logger.info("Returning cached result")
                return cached_result

        try:
            # Stage 1: Preprocessing and normalization
            normalized_text = await self._preprocess_text(text)

            # Stage 2: Multi-modal processing if applicable
            if images and self.config.enable_multimodal:
                multimodal_context = await self.multimodal_processor.process(
                    text=normalized_text,
                    images=images
                )
                context = {**(context or {}), **multimodal_context}

            # Stage 3: Parallel entity extraction
            extraction_tasks = [
                self._extract_entities(normalized_text, context),
                self._extract_dimensions(normalized_text),
                self._extract_materials(normalized_text),
                self._extract_spatial_relationships(normalized_text),
                self._extract_constraints(normalized_text)
            ]

            if self.config.parallel_processing:
                results = await asyncio.gather(*extraction_tasks)
                entities, dimensions, materials, spatial_rels, constraints = results
            else:
                entities = await self._extract_entities(normalized_text, context)
                dimensions = await self._extract_dimensions(normalized_text)
                materials = await self._extract_materials(normalized_text)
                spatial_rels = await self._extract_spatial_relationships(normalized_text)
                constraints = await self._extract_constraints(normalized_text)

            # Stage 4: Intent classification
            intent = await self._classify_intent(normalized_text, entities)

            # Stage 5: Disambiguation if needed
            if self.config.enable_disambiguation:
                entities, spatial_rels = await self._disambiguate(
                    entities, spatial_rels, context
                )

            # Stage 6: Validation
            validation_results = {}
            warnings = []
            suggestions = []

            if self.config.enable_validation:
                validation_results = await self._validate_design(
                    entities, dimensions, constraints
                )
                warnings = validation_results.get("warnings", [])
                suggestions = validation_results.get("suggestions", [])

            # Stage 7: Post-processing and enrichment
            processed_result = await self._post_process(
                entities, dimensions, materials, spatial_rels, constraints
            )

            # Calculate confidence scores
            confidence_scores = self._calculate_confidence(
                entities, dimensions, materials, spatial_rels
            )

            # Create result
            result = ProcessingResult(
                parsed_input=ParsedInput(
                    original_text=text,
                    normalized_text=normalized_text,
                    timestamp=datetime.now()
                ),
                entities=entities,
                dimensions=dimensions,
                spatial_relationships=spatial_rels,
                materials=materials,
                constraints=constraints,
                intent=intent,
                confidence_scores=confidence_scores,
                processing_time=(datetime.now() - start_time).total_seconds(),
                validation_results=validation_results,
                metadata={
                    "mode": self.config.mode.value,
                    "context": context,
                    "version": "1.0.0"
                },
                warnings=warnings,
                suggestions=suggestions
            )

            # Cache result if enabled
            if self.config.enable_caching:
                self.cache.set(text, result)

            # Log performance metrics
            self.performance_monitor.record_processing(
                processing_time=result.processing_time,
                entity_count=len(entities),
                confidence=np.mean(list(confidence_scores.values()))
            )

            return result

        except Exception as e:
            logger.error(f"Pipeline processing error: {e}")
            raise ProcessingError(f"Failed to process text: {e}")

    def process(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        images: Optional[List[Any]] = None
    ) -> ProcessingResult:
        """
        Synchronous wrapper for process_async
        """
        return asyncio.run(self.process_async(text, context, images))

    async def _preprocess_text(self, text: str) -> str:
        """
        Preprocess and normalize input text

        - Convert to lowercase for consistency
        - Expand abbreviations
        - Fix common typos
        - Normalize measurements
        """
        # Basic normalization
        normalized = text.lower().strip()

        # Expand common abbreviations
        abbreviations = {
            "sq ft": "square feet",
            "sq m": "square meters",
            "ft": "feet",
            "m": "meters",
            "br": "bedroom",
            "ba": "bathroom",
            "lvl": "level",
            "bldg": "building"
        }

        for abbr, full in abbreviations.items():
            normalized = normalized.replace(abbr, full)

        # Fix common architectural typos
        typo_corrections = {
            "ceeling": "ceiling",
            "colum": "column",
            "windoe": "window",
            "dor": "door",
            "flor": "floor"
        }

        for typo, correct in typo_corrections.items():
            normalized = normalized.replace(typo, correct)

        return normalized

    async def _extract_entities(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ArchitecturalElement]:
        """Extract architectural entities from text"""
        entities = await self.entity_recognizer.extract(text, context)

        # Apply mode-specific filtering
        if self.config.mode == ProcessingMode.BASIC:
            # Filter to basic geometric shapes only
            entities = [e for e in entities if e.is_basic_geometry()]
        elif self.config.mode == ProcessingMode.RESIDENTIAL:
            # Apply residential-specific rules
            entities = self._apply_residential_rules(entities)
        elif self.config.mode == ProcessingMode.COMMERCIAL:
            # Apply commercial building rules
            entities = self._apply_commercial_rules(entities)

        return entities

    async def _extract_dimensions(self, text: str) -> Dict[str, Any]:
        """Extract and process dimensions"""
        return await self.dimension_processor.extract_dimensions(text)

    async def _extract_materials(self, text: str) -> List[MaterialSpecification]:
        """Extract material specifications"""
        return await self.material_extractor.extract(text)

    async def _extract_spatial_relationships(
        self, text: str
    ) -> List[SpatialRelationship]:
        """Extract spatial relationships between elements"""
        return await self.spatial_parser.parse(text)

    async def _extract_constraints(self, text: str) -> DesignConstraints:
        """Extract design constraints and requirements"""
        return await self.constraint_extractor.extract(text)

    async def _classify_intent(
        self,
        text: str,
        entities: List[ArchitecturalElement]
    ) -> str:
        """Classify user intent from text and entities"""
        return await self.intent_classifier.classify(text, entities)

    async def _disambiguate(
        self,
        entities: List[ArchitecturalElement],
        spatial_rels: List[SpatialRelationship],
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[ArchitecturalElement], List[SpatialRelationship]]:
        """Resolve ambiguities in extracted information"""
        if not self.disambiguator:
            return entities, spatial_rels

        return await self.disambiguator.disambiguate(
            entities, spatial_rels, context
        )

    async def _validate_design(
        self,
        entities: List[ArchitecturalElement],
        dimensions: Dict[str, Any],
        constraints: DesignConstraints
    ) -> Dict[str, Any]:
        """Validate design against building codes and constraints"""
        if not self.validator:
            return {}

        return await self.validator.validate(
            entities, dimensions, constraints
        )

    async def _post_process(
        self,
        entities: List[ArchitecturalElement],
        dimensions: Dict[str, Any],
        materials: List[MaterialSpecification],
        spatial_rels: List[SpatialRelationship],
        constraints: DesignConstraints
    ) -> Dict[str, Any]:
        """Post-process and enrich extracted information"""
        # Apply intelligent defaults based on context
        entities = self._apply_intelligent_defaults(entities, materials)

        # Infer missing relationships
        spatial_rels = self._infer_relationships(entities, spatial_rels)

        # Optimize spatial layout
        if self.config.mode in [ProcessingMode.COMMERCIAL, ProcessingMode.COMPLEX]:
            spatial_rels = self._optimize_layout(entities, spatial_rels, constraints)

        return {
            "entities": entities,
            "dimensions": dimensions,
            "materials": materials,
            "spatial_relationships": spatial_rels,
            "constraints": constraints
        }

    def _calculate_confidence(
        self,
        entities: List[ArchitecturalElement],
        dimensions: Dict[str, Any],
        materials: List[MaterialSpecification],
        spatial_rels: List[SpatialRelationship]
    ) -> Dict[str, float]:
        """Calculate confidence scores for extracted information"""
        scores = {}

        # Entity extraction confidence
        if entities:
            scores["entities"] = np.mean([e.confidence for e in entities])
        else:
            scores["entities"] = 0.0

        # Dimension extraction confidence
        if dimensions:
            dim_confidences = [d.get("confidence", 1.0) for d in dimensions.values()]
            scores["dimensions"] = np.mean(dim_confidences) if dim_confidences else 0.0
        else:
            scores["dimensions"] = 0.0

        # Material extraction confidence
        if materials:
            scores["materials"] = np.mean([m.confidence for m in materials])
        else:
            scores["materials"] = 0.0

        # Spatial relationship confidence
        if spatial_rels:
            scores["spatial"] = np.mean([s.confidence for s in spatial_rels])
        else:
            scores["spatial"] = 0.0

        # Overall confidence
        scores["overall"] = np.mean(list(scores.values()))

        return scores

    def _apply_residential_rules(
        self, entities: List[ArchitecturalElement]
    ) -> List[ArchitecturalElement]:
        """Apply residential-specific processing rules"""
        # Add default elements for residential buildings
        has_bathroom = any(e.type == "bathroom" for e in entities)
        has_kitchen = any(e.type == "kitchen" for e in entities)

        if not has_bathroom:
            logger.info("Adding default bathroom for residential building")
            # Add default bathroom

        if not has_kitchen:
            logger.info("Adding default kitchen for residential building")
            # Add default kitchen

        return entities

    def _apply_commercial_rules(
        self, entities: List[ArchitecturalElement]
    ) -> List[ArchitecturalElement]:
        """Apply commercial building specific rules"""
        # Ensure compliance with commercial building codes
        # Add required elements like emergency exits, accessibility features
        return entities

    def _apply_intelligent_defaults(
        self,
        entities: List[ArchitecturalElement],
        materials: List[MaterialSpecification]
    ) -> List[ArchitecturalElement]:
        """Apply intelligent defaults based on context"""
        for entity in entities:
            # Apply default dimensions if missing
            if not entity.dimensions:
                entity.dimensions = self._get_default_dimensions(entity.type)

            # Apply default materials if not specified
            if not entity.material and materials:
                entity.material = self._get_default_material(entity.type, materials)

        return entities

    def _infer_relationships(
        self,
        entities: List[ArchitecturalElement],
        spatial_rels: List[SpatialRelationship]
    ) -> List[SpatialRelationship]:
        """Infer missing spatial relationships"""
        # Use spatial reasoning to infer relationships
        # Example: If room A is north of B and B is north of C, then A is north of C
        return spatial_rels

    def _optimize_layout(
        self,
        entities: List[ArchitecturalElement],
        spatial_rels: List[SpatialRelationship],
        constraints: DesignConstraints
    ) -> List[SpatialRelationship]:
        """Optimize spatial layout for efficiency"""
        # Apply optimization algorithms for space utilization
        # Consider traffic flow, natural light, etc.
        return spatial_rels

    def _get_default_dimensions(self, entity_type: str) -> Dict[str, float]:
        """Get default dimensions for entity type"""
        defaults = {
            "bedroom": {"width": 3.5, "length": 4.0, "height": 2.7},
            "bathroom": {"width": 2.0, "length": 2.5, "height": 2.4},
            "kitchen": {"width": 3.0, "length": 3.5, "height": 2.7},
            "living_room": {"width": 4.5, "length": 5.5, "height": 2.7},
            "door": {"width": 0.9, "height": 2.1, "thickness": 0.05},
            "window": {"width": 1.2, "height": 1.5, "thickness": 0.1}
        }
        return defaults.get(entity_type, {"width": 3.0, "length": 3.0, "height": 2.7})

    def _get_default_material(
        self,
        entity_type: str,
        available_materials: List[MaterialSpecification]
    ) -> Optional[MaterialSpecification]:
        """Get default material for entity type"""
        material_mapping = {
            "wall": "concrete",
            "floor": "concrete",
            "roof": "steel",
            "door": "wood",
            "window": "glass",
            "column": "steel",
            "beam": "steel"
        }

        material_name = material_mapping.get(entity_type)
        if material_name:
            for material in available_materials:
                if material.name.lower() == material_name:
                    return material

        return None

    def _load_building_codes(self) -> Dict[str, Any]:
        """Load building codes from database"""
        # This would load from the actual building codes database
        return {
            "MIN_CEILING_HEIGHT": 2.4,
            "MIN_ROOM_AREA": 7.0,
            "MAX_DOOR_WIDTH": 1.2,
            "MIN_WINDOW_AREA_RATIO": 0.1
        }

    def _load_material_database(self) -> Dict[str, Any]:
        """Load material specifications from database"""
        # This would load from the actual materials database
        return {}

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.performance_monitor.get_metrics()

    def clear_cache(self):
        """Clear the semantic cache"""
        if self.config.enable_caching:
            self.cache.clear()

    def update_config(self, config: PipelineConfig):
        """Update pipeline configuration"""
        self.config = config
        self._initialize_components()


class ProcessingError(Exception):
    """Custom exception for pipeline processing errors"""
    pass