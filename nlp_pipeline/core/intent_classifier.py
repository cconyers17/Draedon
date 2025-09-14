"""
Intent Classification System for Architectural Commands
Determines user intent from natural language input
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging

logger = logging.getLogger(__name__)


class ArchitecturalIntent(Enum):
    """Types of architectural intents"""
    CREATE = "create"  # Create new structure/element
    MODIFY = "modify"  # Modify existing element
    REMOVE = "remove"  # Remove/delete element
    QUERY = "query"  # Query information
    ANALYZE = "analyze"  # Analyze design
    OPTIMIZE = "optimize"  # Optimize layout/structure
    VALIDATE = "validate"  # Validate against codes
    VISUALIZE = "visualize"  # Generate visualization
    EXPORT = "export"  # Export to CAD format
    COMPARE = "compare"  # Compare designs
    DUPLICATE = "duplicate"  # Copy/duplicate elements
    TRANSFORM = "transform"  # Transform geometry
    CONFIGURE = "configure"  # Configure settings
    SIMULATE = "simulate"  # Run simulation


@dataclass
class IntentClassification:
    """Result of intent classification"""
    primary_intent: ArchitecturalIntent
    confidence: float
    secondary_intents: List[Tuple[ArchitecturalIntent, float]]
    action_keywords: List[str]
    target_elements: List[str]
    modifiers: Dict[str, Any]


class IntentClassifier:
    """
    Advanced intent classification for architectural commands

    Features:
    - Multi-intent detection
    - Context-aware classification
    - Action keyword extraction
    - Confidence scoring
    - Hierarchical intent recognition
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        model_path: Optional[str] = None
    ):
        """Initialize intent classifier"""
        self.confidence_threshold = confidence_threshold

        # Initialize intent patterns
        self._init_intent_patterns()

        # Load or train classification model
        if model_path:
            self._load_model(model_path)
        else:
            self._init_default_classifier()

        logger.info("Intent classifier initialized")

    def _init_intent_patterns(self):
        """Initialize intent keyword patterns"""
        self.intent_keywords = {
            ArchitecturalIntent.CREATE: [
                "create", "build", "add", "construct", "design", "generate",
                "make", "place", "insert", "draw", "establish", "develop",
                "erect", "assemble", "form", "produce"
            ],
            ArchitecturalIntent.MODIFY: [
                "modify", "change", "alter", "adjust", "update", "edit",
                "revise", "transform", "resize", "move", "rotate", "scale",
                "extend", "shorten", "widen", "narrow", "raise", "lower"
            ],
            ArchitecturalIntent.REMOVE: [
                "remove", "delete", "eliminate", "erase", "clear", "demolish",
                "destroy", "take out", "get rid of", "exclude", "omit"
            ],
            ArchitecturalIntent.QUERY: [
                "what", "where", "when", "how", "why", "which", "show",
                "display", "list", "find", "search", "get", "retrieve",
                "tell me", "information about", "details of"
            ],
            ArchitecturalIntent.ANALYZE: [
                "analyze", "evaluate", "assess", "examine", "study", "review",
                "inspect", "investigate", "check", "measure", "calculate",
                "compute", "determine"
            ],
            ArchitecturalIntent.OPTIMIZE: [
                "optimize", "improve", "enhance", "maximize", "minimize",
                "streamline", "refine", "perfect", "tune", "adjust for efficiency"
            ],
            ArchitecturalIntent.VALIDATE: [
                "validate", "verify", "check", "confirm", "ensure", "test",
                "comply", "meet code", "satisfy requirements", "conform"
            ],
            ArchitecturalIntent.VISUALIZE: [
                "visualize", "render", "show", "display", "view", "preview",
                "3d view", "perspective", "elevation", "section", "plan view"
            ],
            ArchitecturalIntent.EXPORT: [
                "export", "save", "output", "generate file", "convert to",
                "download", "extract", "produce drawing"
            ],
            ArchitecturalIntent.COMPARE: [
                "compare", "contrast", "difference", "versus", "vs", "against",
                "side by side", "evaluate options"
            ],
            ArchitecturalIntent.DUPLICATE: [
                "duplicate", "copy", "clone", "replicate", "repeat", "mirror",
                "array", "pattern", "multiply"
            ],
            ArchitecturalIntent.TRANSFORM: [
                "transform", "convert", "translate", "rotate", "flip", "mirror",
                "skew", "deform", "morph", "reshape"
            ],
            ArchitecturalIntent.CONFIGURE: [
                "configure", "set", "setup", "specify", "define", "establish",
                "parameters", "settings", "preferences", "options"
            ],
            ArchitecturalIntent.SIMULATE: [
                "simulate", "model", "test", "run analysis", "predict",
                "forecast", "estimate performance", "structural analysis",
                "thermal analysis", "lighting simulation"
            ]
        }

        # Action verbs for more precise classification
        self.action_verbs = {
            "construction": ["build", "construct", "erect", "assemble"],
            "modification": ["alter", "change", "modify", "adjust"],
            "deletion": ["remove", "delete", "demolish", "eliminate"],
            "information": ["show", "display", "list", "describe"],
            "analysis": ["analyze", "calculate", "evaluate", "assess"],
            "optimization": ["optimize", "improve", "enhance", "refine"]
        }

    def _init_default_classifier(self):
        """Initialize default rule-based classifier"""
        self.use_ml_model = False
        logger.info("Using rule-based intent classification")

    def _load_model(self, model_path: str):
        """Load pre-trained ML model"""
        try:
            self.vectorizer = joblib.load(f"{model_path}_vectorizer.pkl")
            self.classifier = joblib.load(f"{model_path}_classifier.pkl")
            self.use_ml_model = True
            logger.info(f"Loaded ML model from {model_path}")
        except:
            logger.warning("Could not load ML model, falling back to rules")
            self._init_default_classifier()

    async def classify(
        self,
        text: str,
        entities: Optional[List[Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> IntentClassification:
        """
        Classify user intent from text

        Args:
            text: Input text
            entities: Extracted entities (optional)
            context: Additional context (optional)

        Returns:
            IntentClassification with primary and secondary intents
        """
        text_lower = text.lower()

        # Extract action keywords
        action_keywords = self._extract_action_keywords(text_lower)

        # Extract target elements
        target_elements = self._extract_target_elements(text_lower, entities)

        # Get intent scores
        if self.use_ml_model:
            intent_scores = self._classify_with_ml(text_lower)
        else:
            intent_scores = self._classify_with_rules(text_lower, action_keywords)

        # Apply context adjustments
        if context:
            intent_scores = self._adjust_for_context(intent_scores, context)

        # Sort intents by score
        sorted_intents = sorted(
            intent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Get primary intent
        primary_intent = sorted_intents[0][0]
        primary_confidence = sorted_intents[0][1]

        # Get secondary intents above threshold
        secondary_intents = [
            (intent, score)
            for intent, score in sorted_intents[1:]
            if score >= self.confidence_threshold * 0.5
        ]

        # Extract modifiers
        modifiers = self._extract_modifiers(text_lower, primary_intent)

        return IntentClassification(
            primary_intent=primary_intent,
            confidence=primary_confidence,
            secondary_intents=secondary_intents,
            action_keywords=action_keywords,
            target_elements=target_elements,
            modifiers=modifiers
        )

    def _extract_action_keywords(self, text: str) -> List[str]:
        """Extract action keywords from text"""
        keywords = []

        for intent, intent_keywords in self.intent_keywords.items():
            for keyword in intent_keywords:
                if keyword in text:
                    keywords.append(keyword)

        return list(set(keywords))

    def _extract_target_elements(
        self,
        text: str,
        entities: Optional[List[Any]] = None
    ) -> List[str]:
        """Extract target architectural elements"""
        targets = []

        # Common architectural elements
        element_keywords = [
            "wall", "door", "window", "roof", "floor", "ceiling",
            "column", "beam", "room", "building", "structure",
            "staircase", "foundation", "facade", "balcony"
        ]

        for element in element_keywords:
            if element in text:
                targets.append(element)

        # Add entities if provided
        if entities:
            for entity in entities:
                if hasattr(entity, 'name'):
                    targets.append(entity.name)

        return list(set(targets))

    def _classify_with_rules(
        self,
        text: str,
        action_keywords: List[str]
    ) -> Dict[ArchitecturalIntent, float]:
        """Rule-based intent classification"""
        scores = {intent: 0.0 for intent in ArchitecturalIntent}

        # Calculate scores based on keyword matches
        for intent, keywords in self.intent_keywords.items():
            matches = 0
            total_keywords = len(keywords)

            for keyword in keywords:
                if keyword in text:
                    matches += 1
                    # Boost score if keyword appears early in text
                    position_boost = 1.0 - (text.index(keyword) / len(text)) * 0.2
                    scores[intent] += position_boost

            # Normalize score
            if total_keywords > 0:
                scores[intent] = (matches / total_keywords) * (scores[intent] / matches if matches > 0 else 1)

        # Apply rules for specific patterns
        scores = self._apply_pattern_rules(text, scores)

        # Normalize scores to sum to 1
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {intent: score / total_score for intent, score in scores.items()}

        return scores

    def _classify_with_ml(self, text: str) -> Dict[ArchitecturalIntent, float]:
        """ML-based intent classification"""
        # Vectorize text
        features = self.vectorizer.transform([text])

        # Get prediction probabilities
        probabilities = self.classifier.predict_proba(features)[0]

        # Map to intent enum
        scores = {}
        for i, intent in enumerate(ArchitecturalIntent):
            scores[intent] = probabilities[i]

        return scores

    def _apply_pattern_rules(
        self,
        text: str,
        scores: Dict[ArchitecturalIntent, float]
    ) -> Dict[ArchitecturalIntent, float]:
        """Apply specific pattern rules for intent detection"""

        # Questions typically indicate QUERY intent
        if text.strip().endswith("?") or text.startswith(("what", "where", "how", "why")):
            scores[ArchitecturalIntent.QUERY] *= 1.5

        # Imperatives often indicate CREATE or MODIFY
        imperative_verbs = ["make", "build", "create", "add", "place"]
        if any(text.startswith(verb) for verb in imperative_verbs):
            scores[ArchitecturalIntent.CREATE] *= 1.3

        # "Change" or "modify" strongly indicate MODIFY intent
        if "change" in text or "modify" in text or "alter" in text:
            scores[ArchitecturalIntent.MODIFY] *= 1.4

        # "Delete" or "remove" strongly indicate REMOVE intent
        if "delete" in text or "remove" in text:
            scores[ArchitecturalIntent.REMOVE] *= 1.5

        # Technical terms might indicate ANALYZE or VALIDATE
        technical_terms = ["structural", "load", "stress", "compliance", "code"]
        if any(term in text for term in technical_terms):
            scores[ArchitecturalIntent.ANALYZE] *= 1.2
            scores[ArchitecturalIntent.VALIDATE] *= 1.2

        # File format mentions indicate EXPORT
        file_formats = [".dwg", ".dxf", ".step", ".ifc", ".stl", ".obj"]
        if any(fmt in text for fmt in file_formats):
            scores[ArchitecturalIntent.EXPORT] *= 1.5

        return scores

    def _adjust_for_context(
        self,
        scores: Dict[ArchitecturalIntent, float],
        context: Dict[str, Any]
    ) -> Dict[ArchitecturalIntent, float]:
        """Adjust scores based on context"""

        # If in design mode, boost CREATE and MODIFY
        if context.get("mode") == "design":
            scores[ArchitecturalIntent.CREATE] *= 1.2
            scores[ArchitecturalIntent.MODIFY] *= 1.2

        # If in analysis mode, boost ANALYZE and VALIDATE
        elif context.get("mode") == "analysis":
            scores[ArchitecturalIntent.ANALYZE] *= 1.3
            scores[ArchitecturalIntent.VALIDATE] *= 1.3

        # If previous intent was CREATE, likely to be MODIFY next
        if context.get("previous_intent") == ArchitecturalIntent.CREATE:
            scores[ArchitecturalIntent.MODIFY] *= 1.1

        return scores

    def _extract_modifiers(
        self,
        text: str,
        intent: ArchitecturalIntent
    ) -> Dict[str, Any]:
        """Extract intent-specific modifiers"""
        modifiers = {}

        if intent == ArchitecturalIntent.CREATE:
            # Look for creation modifiers
            if "quickly" in text or "fast" in text:
                modifiers["speed"] = "fast"
            if "detailed" in text or "precise" in text:
                modifiers["detail_level"] = "high"
            if "simple" in text or "basic" in text:
                modifiers["complexity"] = "simple"

        elif intent == ArchitecturalIntent.MODIFY:
            # Look for modification type
            if "slightly" in text or "small" in text:
                modifiers["extent"] = "minor"
            if "completely" in text or "major" in text:
                modifiers["extent"] = "major"
            if "resize" in text:
                modifiers["modification_type"] = "resize"
            if "move" in text or "relocate" in text:
                modifiers["modification_type"] = "move"

        elif intent == ArchitecturalIntent.ANALYZE:
            # Look for analysis type
            if "structural" in text:
                modifiers["analysis_type"] = "structural"
            if "thermal" in text or "energy" in text:
                modifiers["analysis_type"] = "thermal"
            if "cost" in text or "budget" in text:
                modifiers["analysis_type"] = "cost"

        elif intent == ArchitecturalIntent.EXPORT:
            # Look for export format
            formats = {
                "dwg": "AutoCAD DWG",
                "dxf": "DXF",
                "step": "STEP",
                "ifc": "IFC",
                "stl": "STL",
                "obj": "OBJ",
                "pdf": "PDF"
            }

            for fmt, name in formats.items():
                if fmt in text.lower():
                    modifiers["format"] = name
                    break

        return modifiers

    def get_intent_description(self, intent: ArchitecturalIntent) -> str:
        """Get human-readable description of intent"""
        descriptions = {
            ArchitecturalIntent.CREATE: "Create new architectural elements",
            ArchitecturalIntent.MODIFY: "Modify existing elements",
            ArchitecturalIntent.REMOVE: "Remove or delete elements",
            ArchitecturalIntent.QUERY: "Query information about the design",
            ArchitecturalIntent.ANALYZE: "Analyze structural or design properties",
            ArchitecturalIntent.OPTIMIZE: "Optimize design for efficiency",
            ArchitecturalIntent.VALIDATE: "Validate against building codes",
            ArchitecturalIntent.VISUALIZE: "Generate visual representation",
            ArchitecturalIntent.EXPORT: "Export to CAD format",
            ArchitecturalIntent.COMPARE: "Compare different design options",
            ArchitecturalIntent.DUPLICATE: "Duplicate or copy elements",
            ArchitecturalIntent.TRANSFORM: "Transform geometric properties",
            ArchitecturalIntent.CONFIGURE: "Configure design settings",
            ArchitecturalIntent.SIMULATE: "Run simulation or analysis"
        }

        return descriptions.get(intent, "Unknown intent")

    def get_suggested_actions(
        self,
        intent: ArchitecturalIntent,
        target_elements: List[str]
    ) -> List[str]:
        """Get suggested actions based on intent and targets"""
        suggestions = []

        if intent == ArchitecturalIntent.CREATE:
            for element in target_elements:
                suggestions.append(f"Create {element} with specified dimensions")
                suggestions.append(f"Add {element} to current design")

        elif intent == ArchitecturalIntent.MODIFY:
            for element in target_elements:
                suggestions.append(f"Resize {element}")
                suggestions.append(f"Move {element} to new position")
                suggestions.append(f"Change {element} material")

        elif intent == ArchitecturalIntent.ANALYZE:
            suggestions.append("Run structural analysis")
            suggestions.append("Calculate load distribution")
            suggestions.append("Evaluate design efficiency")

        return suggestions