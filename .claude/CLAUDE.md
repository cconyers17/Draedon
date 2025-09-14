# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context
Text-to-CAD Architecture Web Application - A comprehensive system that converts natural language architectural descriptions into professional CAD models, supporting both simple residential and complex commercial structures.

## Technical Stack
- **Frontend**: Next.js 14 + React 18 + TypeScript + Three.js for 3D visualization
- **Backend**: Python FastAPI for NLP processing and CAD generation
- **CAD Engine**: OpenCASCADE.js (WebAssembly) for browser-based CAD operations
- **3D APIs**: Meshy AI, Trellis 3D, Rodin AI (using free tiers with fallback strategies)
- **Database**: PostgreSQL for project storage, Redis for caching
- **File Formats**: STEP, IFC, STL, OBJ, DXF export capabilities

## Core Architecture Components

### 1. Natural Language Processing Pipeline
Located in documentation: `text_to_cad_implementation_guide.txt` section 2
- **Architectural Entity Recognition**: Extracts building elements, dimensions, materials, spaces
- **Intent Classification**: CREATE, MODIFY, REMOVE, QUERY operations
- **Constraint Extraction**: Spatial relationships and building code validation
- **Dimension Processing**: Multi-unit support with automatic conversion to meters

### 2. Parametric Design Engine
Referenced in: `computational_design_algorithms.txt`
- **Constraint-Based Modeling**: Geometric relationships and building code compliance
- **L-System Generation**: Algorithmic design generation for complex structures
- **Spatial Layout Generator**: Bubble diagram, grid-based, and tree algorithm approaches
- **Form-Finding Algorithms**: Catenary curves, minimal surfaces, structural optimization

### 3. CAD Geometry Engine
Implementation patterns in: `text_to_cad_implementation_guide.txt` section 3.2
- **B-Rep Operations**: Boolean union, difference, intersection using OpenCASCADE
- **NURBS Implementation**: Parametric curves and surfaces
- **Mesh Generation**: Quality assessment and optimization
- **Multi-format Export**: STEP, IGES, IFC, STL, OBJ, DXF

### 4. Architectural Knowledge Base
Material properties in: `building_materials_database.txt`
Building standards in: `architectural_fundamentals.txt`
- **Material Database**: Comprehensive properties for concrete, steel, wood, glass, composites
- **Building Codes**: IBC compliance, minimum room dimensions, structural requirements
- **Construction Standards**: ISO 19650, IFC specifications, BIM LOD levels

## Development Commands

### Frontend (Next.js)
```bash
# Install dependencies
npm install

# Development server
npm run dev

# Build for production
npm run build

# Type checking
npm run type-check

# Linting
npm run lint
```

### Backend (Python FastAPI)
```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn main:app --reload --port 8000

# Run tests
pytest

# Type checking
mypy .
```

## Complexity Levels (L0-L3)

### L0 - Basic Geometric Shapes
- Simple rectangles, circles, basic extrusions
- Single-material structures
- Basic room layouts

### L1 - Residential Buildings
- Multi-room layouts with proper adjacencies
- Standard construction materials
- Building code validation for residential

### L2 - Commercial Buildings
- Complex spatial relationships
- Multiple building systems integration
- Advanced structural calculations

### L3 - Iconic Structures (Oculus, The Line Dubai)
- Custom geometric algorithms
- Advanced parametric relationships
- Complex environmental systems integration

## Key Algorithms and Implementations

### Entity Recognition Patterns
```python
entity_types = {
    'BUILDING_ELEMENT': ['wall', 'door', 'window', 'roof', 'floor', 'column', 'beam'],
    'DIMENSION': ['length', 'width', 'height', 'thickness', 'diameter'],
    'MATERIAL': ['concrete', 'steel', 'wood', 'glass', 'brick'],
    'SPACE': ['room', 'office', 'bathroom', 'kitchen', 'hallway'],
    'LOCATION': ['north', 'south', 'east', 'west', 'center', 'corner'],
    'QUANTITY': ['one', 'two', 'three', 'multiple', 'several']
}
```

### Building Code Constraints
```python
building_codes = {
    'MIN_CEILING_HEIGHT': 2.4,      # meters
    'MIN_ROOM_AREA': 7.0,           # square meters
    'MAX_DOOR_WIDTH': 1.2,          # meters
    'MIN_WINDOW_AREA_RATIO': 0.1    # 10% of floor area
}
```

### Material Properties Database
Reference comprehensive material specifications including:
- Density, strength properties
- Thermal conductivity
- Cost per unit
- Carbon footprint
- Fire rating

## API Integration Strategy

### Free Tier Usage
- **Meshy AI**: 200 free generations/month
- **Trellis 3D**: Academic/research tier
- **Rodin AI**: Limited free credits

### Fallback Strategies
1. Local geometry generation using OpenCASCADE.js
2. Simplified mesh generation for complex models
3. Progressive detail enhancement based on API availability

## File Structure Conventions
```
src/
├── components/          # React components
│   ├── ui/             # Reusable UI components
│   ├── cad/            # CAD-specific components
│   └── visualization/  # Three.js components
├── lib/                # Utility libraries
│   ├── nlp/           # Natural language processing
│   ├── cad/           # CAD operations
│   └── materials/     # Material database
├── pages/             # Next.js pages
├── api/               # API routes
└── types/             # TypeScript definitions
```

## Testing Strategy
- Unit tests for NLP components
- Integration tests for CAD generation pipeline
- End-to-end tests for complete text-to-CAD workflow
- Visual regression tests for 3D rendering

## Deployment Considerations
- WebAssembly module loading for OpenCASCADE.js
- API rate limiting and fallback handling
- Large file export optimization
- Browser compatibility for 3D features

## Documentation References
Always reference the comprehensive documentation files:
- `architectural_fundamentals.txt`: Building codes, design principles
- `building_materials_database.txt`: Material properties and specifications
- `cad_technical_specifications.txt`: CAD file formats and standards
- `computational_design_algorithms.txt`: Algorithmic design implementations
- `text_to_cad_implementation_guide.txt`: Complete implementation patterns