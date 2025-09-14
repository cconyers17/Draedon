"""
Advanced Architectural Entity Recognition System
Uses transformer models and custom NER for architectural element extraction
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    pipeline as hf_pipeline
)
import spacy
from spacy.tokens import Span
from spacy.matcher import Matcher, PhraseMatcher
import logging

from ..models.architectural_models import ArchitecturalElement, ElementType
from ..utils.patterns import ARCHITECTURAL_PATTERNS
from ..utils.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


class EntityCategory(Enum):
    """Categories of architectural entities"""
    STRUCTURAL = "structural"  # Walls, columns, beams, foundations
    SPATIAL = "spatial"  # Rooms, spaces, zones
    OPENING = "opening"  # Doors, windows, skylights
    SYSTEM = "system"  # HVAC, electrical, plumbing
    MATERIAL = "material"  # Concrete, steel, wood, glass
    FURNITURE = "furniture"  # Built-in furniture, fixtures
    LANDSCAPE = "landscape"  # Gardens, courtyards, terraces
    DIMENSION = "dimension"  # Measurements, sizes
    MODIFIER = "modifier"  # Descriptive attributes


@dataclass
class EntityMatch:
    """Represents a matched entity in text"""
    text: str
    category: EntityCategory
    type: str
    start_pos: int
    end_pos: int
    confidence: float
    attributes: Dict[str, Any]
    context: Optional[str] = None


class ArchitecturalEntityRecognizer:
    """
    Advanced entity recognition for architectural text

    Features:
    - Transformer-based NER with fine-tuned models
    - Rule-based pattern matching for domain-specific terms
    - Hierarchical entity recognition
    - Context-aware entity resolution
    - Multi-lingual support
    """

    def __init__(
        self,
        model_version: str = "latest",
        use_gpu: bool = True,
        language: str = "en"
    ):
        """Initialize the entity recognizer"""
        self.model_version = model_version
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.language = language

        # Initialize models
        self._initialize_models()

        # Load architectural patterns
        self._load_patterns()

        # Initialize embedding model for semantic similarity
        self.embedding_model = EmbeddingModel()

        logger.info(f"Entity recognizer initialized with GPU: {self.use_gpu}")

    def _initialize_models(self):
        """Initialize NLP models"""
        # Load spaCy model for linguistic features
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except:
            # Fallback to smaller model
            self.nlp = spacy.load("en_core_web_sm")

        # Add custom components to spaCy pipeline
        self._add_custom_components()

        # Load transformer model for advanced NER
        self._load_transformer_model()

        # Initialize matchers
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab)

    def _load_transformer_model(self):
        """Load fine-tuned transformer model for architectural NER"""
        try:
            # Load custom fine-tuned model if available
            model_name = f"architectural-ner-{self.model_version}"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.ner_model = AutoModelForTokenClassification.from_pretrained(model_name)

            if self.use_gpu:
                self.ner_model = self.ner_model.cuda()

            self.ner_pipeline = hf_pipeline(
                "ner",
                model=self.ner_model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=0 if self.use_gpu else -1
            )
        except:
            # Fallback to base model
            logger.warning("Custom model not found, using base BERT model")
            self.ner_pipeline = None

    def _add_custom_components(self):
        """Add custom components to spaCy pipeline"""
        # Add architectural entity recognizer
        @spacy.Language.component("architectural_ner")
        def architectural_ner(doc):
            entities = self._detect_architectural_entities(doc)
            doc.ents = entities
            return doc

        if "architectural_ner" not in self.nlp.pipe_names:
            self.nlp.add_pipe("architectural_ner", last=True)

    def _load_patterns(self):
        """Load architectural entity patterns"""
        self.patterns = {
            EntityCategory.STRUCTURAL: {
                "wall": ["wall", "partition", "facade", "exterior wall", "interior wall",
                        "load-bearing wall", "curtain wall", "retaining wall"],
                "column": ["column", "pillar", "post", "support", "pier"],
                "beam": ["beam", "girder", "joist", "lintel", "header"],
                "slab": ["slab", "floor slab", "roof slab", "concrete slab"],
                "foundation": ["foundation", "footing", "pile", "basement"],
                "roof": ["roof", "roofing", "pitched roof", "flat roof", "gable roof",
                        "hip roof", "shed roof", "mansard roof"],
                "staircase": ["staircase", "stairs", "stairwell", "steps", "riser", "tread"]
            },
            EntityCategory.SPATIAL: {
                "room": ["room", "space", "area", "zone", "chamber"],
                "bedroom": ["bedroom", "master bedroom", "guest bedroom", "sleeping area"],
                "bathroom": ["bathroom", "restroom", "washroom", "powder room", "ensuite"],
                "kitchen": ["kitchen", "kitchenette", "cooking area", "pantry"],
                "living_room": ["living room", "lounge", "sitting room", "family room",
                               "great room", "den"],
                "office": ["office", "study", "workspace", "home office", "workroom"],
                "hallway": ["hallway", "corridor", "passage", "gallery", "foyer",
                           "entrance hall", "vestibule"],
                "garage": ["garage", "carport", "parking space", "car park"],
                "balcony": ["balcony", "terrace", "deck", "patio", "veranda", "loggia"]
            },
            EntityCategory.OPENING: {
                "door": ["door", "doorway", "entrance", "exit", "portal", "french door",
                        "sliding door", "revolving door", "garage door"],
                "window": ["window", "fenestration", "glazing", "casement window",
                          "sliding window", "bay window", "dormer window", "skylight"],
                "opening": ["opening", "aperture", "void", "hole", "gap"]
            },
            EntityCategory.SYSTEM: {
                "hvac": ["hvac", "heating", "cooling", "ventilation", "air conditioning",
                        "heat pump", "furnace", "boiler", "radiator"],
                "electrical": ["electrical", "wiring", "lighting", "outlet", "switch",
                             "circuit breaker", "panel", "conduit"],
                "plumbing": ["plumbing", "pipes", "water supply", "drainage", "sewage",
                           "fixtures", "faucet", "toilet", "shower"],
                "elevator": ["elevator", "lift", "escalator", "moving walkway"]
            },
            EntityCategory.MATERIAL: {
                "concrete": ["concrete", "reinforced concrete", "precast concrete", "cement"],
                "steel": ["steel", "structural steel", "rebar", "metal", "iron"],
                "wood": ["wood", "timber", "lumber", "plywood", "hardwood", "softwood"],
                "glass": ["glass", "glazing", "tempered glass", "laminated glass",
                         "insulated glass", "stained glass"],
                "brick": ["brick", "masonry", "block", "stone", "marble", "granite"],
                "insulation": ["insulation", "thermal insulation", "acoustic insulation",
                             "foam", "fiberglass", "mineral wool"]
            },
            EntityCategory.FURNITURE: {
                "cabinet": ["cabinet", "cupboard", "closet", "wardrobe", "shelving"],
                "counter": ["counter", "countertop", "worktop", "bench", "island"],
                "fixture": ["fixture", "fitting", "appliance", "equipment"]
            },
            EntityCategory.LANDSCAPE: {
                "garden": ["garden", "yard", "lawn", "landscape", "greenspace"],
                "pool": ["pool", "swimming pool", "spa", "hot tub", "water feature"],
                "driveway": ["driveway", "drive", "access road", "pathway", "walkway"]
            }
        }

        # Create pattern matchers
        self._create_pattern_matchers()

    def _create_pattern_matchers(self):
        """Create spaCy pattern matchers for architectural terms"""
        for category, terms_dict in self.patterns.items():
            for entity_type, terms in terms_dict.items():
                # Create patterns for exact matches
                patterns = [[{"LOWER": term.lower()}] for term in terms]

                # Add pattern to matcher
                self.matcher.add(
                    f"{category.value}_{entity_type}",
                    patterns
                )

                # Add phrases to phrase matcher
                phrase_patterns = [self.nlp.make_doc(term) for term in terms]
                self.phrase_matcher.add(
                    f"PHRASE_{category.value}_{entity_type}",
                    phrase_patterns
                )

    async def extract(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ArchitecturalElement]:
        """
        Extract architectural entities from text

        Args:
            text: Input text to process
            context: Optional context information

        Returns:
            List of extracted architectural elements
        """
        # Process text with spaCy
        doc = self.nlp(text)

        # Extract entities using multiple methods
        entities = []

        # Method 1: Transformer-based NER
        if self.ner_pipeline:
            transformer_entities = self._extract_with_transformer(text)
            entities.extend(transformer_entities)

        # Method 2: Rule-based pattern matching
        pattern_entities = self._extract_with_patterns(doc)
        entities.extend(pattern_entities)

        # Method 3: Dependency parsing for complex entities
        dependency_entities = self._extract_with_dependencies(doc)
        entities.extend(dependency_entities)

        # Method 4: Contextual entity extraction
        if context:
            contextual_entities = self._extract_with_context(doc, context)
            entities.extend(contextual_entities)

        # Merge and deduplicate entities
        merged_entities = self._merge_entities(entities)

        # Resolve entity references
        resolved_entities = self._resolve_references(merged_entities, doc)

        # Extract attributes for each entity
        enriched_entities = self._extract_attributes(resolved_entities, doc)

        # Convert to ArchitecturalElement objects
        architectural_elements = self._convert_to_elements(enriched_entities)

        return architectural_elements

    def _extract_with_transformer(self, text: str) -> List[EntityMatch]:
        """Extract entities using transformer model"""
        if not self.ner_pipeline:
            return []

        entities = []
        try:
            # Run NER pipeline
            ner_results = self.ner_pipeline(text)

            for result in ner_results:
                # Map transformer labels to our categories
                category = self._map_label_to_category(result["entity_group"])
                if category:
                    entities.append(EntityMatch(
                        text=result["word"],
                        category=category,
                        type=result["entity_group"],
                        start_pos=result["start"],
                        end_pos=result["end"],
                        confidence=result["score"],
                        attributes={}
                    ))
        except Exception as e:
            logger.error(f"Transformer extraction error: {e}")

        return entities

    def _extract_with_patterns(self, doc) -> List[EntityMatch]:
        """Extract entities using pattern matching"""
        entities = []

        # Use Matcher for patterns
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            match_label = self.nlp.vocab.strings[match_id]

            # Parse category and type from label
            parts = match_label.split("_")
            if len(parts) >= 2:
                category = EntityCategory(parts[0])
                entity_type = "_".join(parts[1:])

                entities.append(EntityMatch(
                    text=span.text,
                    category=category,
                    type=entity_type,
                    start_pos=span.start_char,
                    end_pos=span.end_char,
                    confidence=0.95,  # High confidence for exact matches
                    attributes={}
                ))

        # Use PhraseMatcher for multi-word patterns
        phrase_matches = self.phrase_matcher(doc)
        for match_id, start, end in phrase_matches:
            span = doc[start:end]
            match_label = self.nlp.vocab.strings[match_id]

            if match_label.startswith("PHRASE_"):
                parts = match_label[7:].split("_")
                if len(parts) >= 2:
                    category = EntityCategory(parts[0])
                    entity_type = "_".join(parts[1:])

                    entities.append(EntityMatch(
                        text=span.text,
                        category=category,
                        type=entity_type,
                        start_pos=span.start_char,
                        end_pos=span.end_char,
                        confidence=0.95,
                        attributes={}
                    ))

        return entities

    def _extract_with_dependencies(self, doc) -> List[EntityMatch]:
        """Extract entities using dependency parsing"""
        entities = []

        # Look for compound noun phrases that might be architectural elements
        for token in doc:
            if token.dep_ == "compound" and token.head.pos_ == "NOUN":
                # Check if this could be an architectural term
                compound_phrase = f"{token.text} {token.head.text}"
                category = self._classify_compound_phrase(compound_phrase)

                if category:
                    entities.append(EntityMatch(
                        text=compound_phrase,
                        category=category,
                        type=self._get_entity_type(compound_phrase, category),
                        start_pos=token.idx,
                        end_pos=token.head.idx + len(token.head.text),
                        confidence=0.85,
                        attributes={}
                    ))

        # Look for entities with modifiers
        for chunk in doc.noun_chunks:
            # Check if chunk contains architectural terms
            if self._is_architectural_chunk(chunk):
                category = self._classify_text(chunk.text)
                if category:
                    entities.append(EntityMatch(
                        text=chunk.text,
                        category=category,
                        type=self._get_entity_type(chunk.root.text, category),
                        start_pos=chunk.start_char,
                        end_pos=chunk.end_char,
                        confidence=0.8,
                        attributes=self._extract_modifiers(chunk)
                    ))

        return entities

    def _extract_with_context(
        self, doc, context: Dict[str, Any]
    ) -> List[EntityMatch]:
        """Extract entities using contextual information"""
        entities = []

        # Use context to identify implied entities
        if "building_type" in context:
            building_type = context["building_type"]
            implied_entities = self._get_implied_entities(building_type)

            for entity_type, attributes in implied_entities.items():
                # Check if entity is mentioned in text
                if self._is_entity_referenced(doc, entity_type):
                    entities.append(EntityMatch(
                        text=entity_type,
                        category=attributes["category"],
                        type=entity_type,
                        start_pos=0,
                        end_pos=0,
                        confidence=0.7,
                        attributes=attributes.get("defaults", {})
                    ))

        return entities

    def _merge_entities(self, entities: List[EntityMatch]) -> List[EntityMatch]:
        """Merge and deduplicate extracted entities"""
        merged = {}

        for entity in entities:
            key = (entity.text.lower(), entity.category, entity.type)

            if key in merged:
                # Keep entity with higher confidence
                if entity.confidence > merged[key].confidence:
                    merged[key] = entity
                else:
                    # Merge attributes
                    merged[key].attributes.update(entity.attributes)
            else:
                merged[key] = entity

        return list(merged.values())

    def _resolve_references(
        self, entities: List[EntityMatch], doc
    ) -> List[EntityMatch]:
        """Resolve entity references and coreferences"""
        resolved = []

        # Build entity index
        entity_index = {e.text.lower(): e for e in entities}

        for entity in entities:
            # Check for references like "the wall", "that room"
            if entity.text.lower().startswith(("the ", "that ", "this ")):
                # Try to resolve reference
                base_text = entity.text[4:].strip()
                if base_text in entity_index:
                    # Merge with referenced entity
                    referenced = entity_index[base_text]
                    entity.attributes.update(referenced.attributes)

            resolved.append(entity)

        return resolved

    def _extract_attributes(
        self, entities: List[EntityMatch], doc
    ) -> List[EntityMatch]:
        """Extract attributes for each entity"""
        for entity in entities:
            # Find entity span in doc
            span = self._find_span(doc, entity.start_pos, entity.end_pos)
            if span:
                # Extract dimensions
                dimensions = self._extract_dimensions_near_entity(doc, span)
                if dimensions:
                    entity.attributes["dimensions"] = dimensions

                # Extract materials
                materials = self._extract_materials_near_entity(doc, span)
                if materials:
                    entity.attributes["materials"] = materials

                # Extract spatial relationships
                spatial = self._extract_spatial_info(doc, span)
                if spatial:
                    entity.attributes["spatial"] = spatial

                # Extract modifiers and adjectives
                modifiers = self._extract_entity_modifiers(span)
                if modifiers:
                    entity.attributes["modifiers"] = modifiers

        return entities

    def _convert_to_elements(
        self, entities: List[EntityMatch]
    ) -> List[ArchitecturalElement]:
        """Convert EntityMatch objects to ArchitecturalElement"""
        elements = []

        for entity in entities:
            element = ArchitecturalElement(
                type=ElementType.from_string(entity.type),
                name=entity.text,
                category=entity.category.value,
                confidence=entity.confidence,
                dimensions=entity.attributes.get("dimensions"),
                material=entity.attributes.get("materials"),
                properties=entity.attributes,
                position={"start": entity.start_pos, "end": entity.end_pos}
            )
            elements.append(element)

        return elements

    def _classify_compound_phrase(self, phrase: str) -> Optional[EntityCategory]:
        """Classify a compound phrase into entity category"""
        phrase_lower = phrase.lower()

        for category, terms_dict in self.patterns.items():
            for entity_type, terms in terms_dict.items():
                if any(term in phrase_lower for term in terms):
                    return category

        # Use semantic similarity if no exact match
        if self.embedding_model:
            best_category = self._classify_by_similarity(phrase)
            if best_category:
                return best_category

        return None

    def _classify_by_similarity(self, text: str) -> Optional[EntityCategory]:
        """Classify text using semantic similarity"""
        # Get embedding for input text
        text_embedding = self.embedding_model.encode(text)

        best_similarity = 0
        best_category = None

        for category, terms_dict in self.patterns.items():
            # Get embeddings for category terms
            all_terms = []
            for terms in terms_dict.values():
                all_terms.extend(terms)

            if all_terms:
                term_embeddings = self.embedding_model.encode(all_terms)
                # Calculate similarity
                similarities = self.embedding_model.similarity(
                    text_embedding, term_embeddings
                )
                max_similarity = np.max(similarities)

                if max_similarity > best_similarity and max_similarity > 0.7:
                    best_similarity = max_similarity
                    best_category = category

        return best_category

    def _is_architectural_chunk(self, chunk) -> bool:
        """Check if noun chunk is architectural"""
        chunk_text = chunk.text.lower()

        # Check against all patterns
        for terms_dict in self.patterns.values():
            for terms in terms_dict.values():
                if any(term in chunk_text for term in terms):
                    return True

        return False

    def _classify_text(self, text: str) -> Optional[EntityCategory]:
        """Classify text into entity category"""
        return self._classify_compound_phrase(text)

    def _get_entity_type(self, text: str, category: EntityCategory) -> str:
        """Get specific entity type from text and category"""
        text_lower = text.lower()

        if category in self.patterns:
            for entity_type, terms in self.patterns[category].items():
                if any(term in text_lower for term in terms):
                    return entity_type

        return text_lower.replace(" ", "_")

    def _extract_modifiers(self, chunk) -> Dict[str, Any]:
        """Extract modifiers from noun chunk"""
        modifiers = {
            "adjectives": [],
            "quantifiers": [],
            "determiners": []
        }

        for token in chunk:
            if token.pos_ == "ADJ":
                modifiers["adjectives"].append(token.text)
            elif token.pos_ == "NUM":
                modifiers["quantifiers"].append(token.text)
            elif token.pos_ == "DET":
                modifiers["determiners"].append(token.text)

        return modifiers

    def _get_implied_entities(
        self, building_type: str
    ) -> Dict[str, Dict[str, Any]]:
        """Get implied entities based on building type"""
        implied = {}

        if building_type == "residential":
            implied = {
                "foundation": {
                    "category": EntityCategory.STRUCTURAL,
                    "defaults": {"type": "concrete", "depth": 1.5}
                },
                "roof": {
                    "category": EntityCategory.STRUCTURAL,
                    "defaults": {"type": "pitched", "material": "tiles"}
                }
            }
        elif building_type == "commercial":
            implied = {
                "elevator": {
                    "category": EntityCategory.SYSTEM,
                    "defaults": {"type": "passenger", "capacity": 10}
                },
                "fire_exit": {
                    "category": EntityCategory.OPENING,
                    "defaults": {"width": 1.2, "count": 2}
                }
            }

        return implied

    def _is_entity_referenced(self, doc, entity_type: str) -> bool:
        """Check if entity type is referenced in document"""
        entity_lower = entity_type.lower()
        doc_lower = doc.text.lower()
        return entity_lower in doc_lower

    def _find_span(self, doc, start_pos: int, end_pos: int):
        """Find span in document by character positions"""
        for sent in doc.sents:
            if sent.start_char <= start_pos and sent.end_char >= end_pos:
                for token in sent:
                    if token.idx == start_pos:
                        start_token = token
                        for end_token in sent:
                            if end_token.idx + len(end_token.text) == end_pos:
                                return doc[start_token.i:end_token.i + 1]
        return None

    def _extract_dimensions_near_entity(self, doc, span) -> Dict[str, float]:
        """Extract dimensions mentioned near an entity"""
        dimensions = {}

        # Look for measurements in same sentence
        sent = span.sent

        # Pattern for dimensions (e.g., "3m x 4m", "10 feet wide")
        dimension_pattern = r'(\d+(?:\.\d+)?)\s*(?:m|meter|metres?|ft|feet|\'|")'

        matches = re.finditer(dimension_pattern, sent.text)
        for match in matches:
            value = float(match.group(1))
            # Determine dimension type based on context
            if "width" in sent.text.lower() or "wide" in sent.text.lower():
                dimensions["width"] = value
            elif "height" in sent.text.lower() or "tall" in sent.text.lower():
                dimensions["height"] = value
            elif "length" in sent.text.lower() or "long" in sent.text.lower():
                dimensions["length"] = value
            elif "x" in match.string[match.start():match.end() + 10]:
                # Handle "3m x 4m" pattern
                if "width" not in dimensions:
                    dimensions["width"] = value
                elif "length" not in dimensions:
                    dimensions["length"] = value

        return dimensions

    def _extract_materials_near_entity(self, doc, span) -> List[str]:
        """Extract materials mentioned near an entity"""
        materials = []

        # Look in same sentence
        sent = span.sent

        for category, terms_dict in self.patterns.items():
            if category == EntityCategory.MATERIAL:
                for material_type, terms in terms_dict.items():
                    for term in terms:
                        if term.lower() in sent.text.lower():
                            materials.append(material_type)

        return materials

    def _extract_spatial_info(self, doc, span) -> Dict[str, Any]:
        """Extract spatial information related to entity"""
        spatial = {}

        sent = span.sent

        # Directional terms
        directions = ["north", "south", "east", "west", "center", "corner",
                     "left", "right", "front", "back", "top", "bottom"]

        for direction in directions:
            if direction in sent.text.lower():
                spatial["direction"] = direction

        # Relative positions
        prepositions = ["next to", "adjacent to", "beside", "above", "below",
                       "between", "opposite", "facing", "near", "far from"]

        for prep in prepositions:
            if prep in sent.text.lower():
                spatial["relation"] = prep

        return spatial

    def _extract_entity_modifiers(self, span) -> List[str]:
        """Extract modifying words for an entity"""
        modifiers = []

        for token in span:
            if token.dep_ in ["amod", "advmod", "compound"]:
                modifiers.append(token.text)

        # Also check tokens immediately before span
        if span.start > 0:
            prev_token = span.doc[span.start - 1]
            if prev_token.dep_ == "amod":
                modifiers.append(prev_token.text)

        return modifiers

    def _map_label_to_category(self, label: str) -> Optional[EntityCategory]:
        """Map transformer model label to entity category"""
        label_mapping = {
            "STRUCTURE": EntityCategory.STRUCTURAL,
            "SPACE": EntityCategory.SPATIAL,
            "OPENING": EntityCategory.OPENING,
            "SYSTEM": EntityCategory.SYSTEM,
            "MATERIAL": EntityCategory.MATERIAL,
            "FURNITURE": EntityCategory.FURNITURE,
            "LANDSCAPE": EntityCategory.LANDSCAPE
        }

        return label_mapping.get(label.upper())