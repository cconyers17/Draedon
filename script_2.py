# Create comprehensive building materials and properties documentation
building_materials_database = """
# COMPREHENSIVE BUILDING MATERIALS DATABASE

## 1. STRUCTURAL MATERIALS

### 1.1 CONCRETE MATERIALS

#### Standard Portland Cement Concrete
**Compressive Strength:** 20-40 MPa (2,900-5,800 psi)
**Tensile Strength:** 2-5 MPa (290-725 psi)
**Density:** 2,200-2,500 kg/m³ (137-156 lb/ft³)
**Elastic Modulus:** 20-40 GPa (2.9-5.8 × 10⁶ psi)
**Thermal Conductivity:** 1.4-2.9 W/mK
**Thermal Expansion:** 10-14 × 10⁻⁶ /K
**Fire Resistance:** Excellent (up to 4 hours)
**Durability:** 50-100+ years

**Mix Design Components:**
- Portland cement: 10-15%
- Water: 4-8%
- Fine aggregate (sand): 25-30%
- Coarse aggregate (gravel): 50-60%
- Air: 1-3%

#### High-Strength Concrete
**Compressive Strength:** 40-100+ MPa (5,800-14,500+ psi)
**Applications:** High-rise buildings, bridges, prestressed elements
**Special additives:** Silica fume, fly ash, superplasticizers
**Water-cement ratio:** <0.35
**Curing requirements:** Extended curing period, controlled temperature

#### Reinforced Concrete Systems
**Steel reinforcement grades:**
- Grade 60: 420 MPa (60,000 psi) yield strength
- Grade 75: 520 MPa (75,000 psi) yield strength
- Grade 80: 550 MPa (80,000 psi) yield strength

**Rebar specifications:**
- #3 (10M): 9.5mm diameter, 0.11 in²
- #4 (15M): 12.7mm diameter, 0.20 in²
- #5 (15M): 15.9mm diameter, 0.31 in²
- #6 (20M): 19.1mm diameter, 0.44 in²
- #8 (25M): 25.4mm diameter, 0.79 in²

#### Specialized Concrete Types

**Lightweight Concrete**
- Density: 1,400-1,900 kg/m³ (87-119 lb/ft³)
- Compressive strength: 15-40 MPa (2,200-5,800 psi)
- Thermal conductivity: 0.4-1.2 W/mK
- Applications: Deck slabs, insulating concrete, precast panels

**Glass Fiber Reinforced Concrete (GFRC)**
- Fiber content: 3-5% by weight
- Flexural strength: 7-25 MPa (1,000-3,600 psi)
- Impact resistance: High
- Applications: Architectural panels, facades, complex shapes

**Translucent Concrete**
- Light transmission: 3-6%
- Optical fiber content: 4-5% by volume
- Compressive strength: 30-50 MPa (4,350-7,250 psi)
- Applications: Facade elements, artistic installations

### 1.2 STEEL MATERIALS

#### Structural Steel Grades

**ASTM A36 Steel**
- Yield strength: 250 MPa (36,000 psi)
- Tensile strength: 400-550 MPa (58,000-80,000 psi)
- Elastic modulus: 200 GPa (29 × 10⁶ psi)
- Density: 7,850 kg/m³ (490 lb/ft³)
- Applications: General construction, low-rise buildings

**ASTM A992 Steel (Wide Flange Shapes)**
- Yield strength: 345-450 MPa (50,000-65,000 psi)
- Tensile strength: 450 MPa minimum (65,000 psi)
- Applications: High-rise construction, heavy industrial buildings

**ASTM A572 High-Strength Steel**
- Grade 50: Yield strength 345 MPa (50,000 psi)
- Grade 65: Yield strength 450 MPa (65,000 psi)
- Applications: Bridges, high-rise buildings, heavy construction

#### Steel Section Properties

**Wide Flange Sections (W-shapes)**
```
W14x120: 14" nominal depth, 120 lb/ft
- Depth: 14.48"
- Flange width: 14.670"
- Web thickness: 0.590"
- Flange thickness: 0.940"
- Area: 35.3 in²
- Moment of inertia: 1,380 in⁴

W24x68: 24" nominal depth, 68 lb/ft
- Depth: 23.73"
- Flange width: 8.965"
- Web thickness: 0.415"
- Flange thickness: 0.585"
- Area: 20.1 in²
- Moment of inertia: 1,830 in⁴
```

**Steel Tube Sections (HSS - Hollow Structural Sections)**
```
HSS12x12x1/2: 12"x12" square tube, 1/2" wall
- Outside dimensions: 12.000" × 12.000"
- Wall thickness: 0.500"
- Area: 21.6 in²
- Moment of inertia: 694 in⁴

HSS8x6x3/8: 8"x6" rectangular tube, 3/8" wall
- Outside dimensions: 8.000" × 6.000"
- Wall thickness: 0.375"
- Area: 9.58 in²
- Moment of inertia: 72.6 in⁴ (strong axis)
```

#### Stainless Steel Properties

**304 Stainless Steel**
- Yield strength: 205-310 MPa (30,000-45,000 psi)
- Tensile strength: 515-620 MPa (75,000-90,000 psi)
- Corrosion resistance: Excellent
- Applications: Architectural cladding, handrails, fixtures

**316 Stainless Steel**
- Yield strength: 205-310 MPa (30,000-45,000 psi)
- Tensile strength: 515-620 MPa (75,000-90,000 psi)
- Enhanced corrosion resistance (marine environments)
- Applications: Coastal structures, chemical processing facilities

### 1.3 WOOD MATERIALS

#### Structural Lumber Classifications

**Douglas Fir-Larch**
- Density: 540 kg/m³ (34 lb/ft³)
- Compressive strength: 50.3 MPa (7,300 psi)
- Flexural strength: 85.5 MPa (12,400 psi)
- Elastic modulus: 13.1 GPa (1.9 × 10⁶ psi)
- Moisture content: 19% maximum

**Southern Pine**
- Density: 580 kg/m³ (36 lb/ft³)
- Compressive strength: 55.2 MPa (8,000 psi)
- Flexural strength: 91.0 MPa (13,200 psi)
- Elastic modulus: 14.5 GPa (2.1 × 10⁶ psi)

#### Engineered Lumber Products

**Glued Laminated Timber (Glulam)**
- Beam depths: 89mm to 1,800mm (3.5" to 71")
- Beam widths: 80mm, 130mm, 175mm (3.1", 5.1", 6.9")
- Flexural strength: 16.5-24 MPa (2,400-3,500 psi)
- Applications: Long-span beams, arches, complex shapes

**Laminated Veneer Lumber (LVL)**
- Standard depths: 241mm, 302mm, 356mm, 406mm (9.5", 11.9", 14", 16")
- Standard widths: 38mm, 45mm (1.5", 1.75")
- Flexural strength: 19.3 MPa (2,800 psi)
- Applications: Headers, beams, scaffold planking

**Cross-Laminated Timber (CLT)**
- Panel thicknesses: 60mm to 400mm (2.4" to 15.75")
- Panel widths: Up to 3.5m (11.5')
- Panel lengths: Up to 20m (65')
- Applications: Floor slabs, wall panels, roof decks

## 2. ENVELOPE MATERIALS

### 2.1 GLASS SYSTEMS

#### Standard Float Glass
**Thickness options:** 3mm, 4mm, 5mm, 6mm, 8mm, 10mm, 12mm, 15mm, 19mm
**Density:** 2,500 kg/m³ (156 lb/ft³)
**Thermal conductivity:** 1.0 W/mK
**Solar heat gain coefficient:** 0.86
**Visible light transmittance:** 90%
**U-value:** 5.8 W/m²K (single pane)

#### Low-E Glass Systems
**Hard coat (pyrolytic) Low-E:**
- Emissivity: 0.15-0.20
- Solar heat gain coefficient: 0.70-0.75
- Visible light transmittance: 78-88%
- Applications: Climate zones with heating concerns

**Soft coat (sputtered) Low-E:**
- Emissivity: 0.04-0.10
- Solar heat gain coefficient: 0.27-0.40
- Visible light transmittance: 70-80%
- Applications: Climate zones with cooling concerns

#### Insulated Glass Unit (IGU) Performance
**Double-glazed standard:**
- U-value: 2.8 W/m²K (0.49 Btu/hr·ft²·°F)
- Air space: 12mm or 16mm (1/2" or 5/8")
- SHGC: 0.76 (clear glass)

**Double-glazed Low-E with argon:**
- U-value: 1.4-1.8 W/m²K (0.25-0.32 Btu/hr·ft²·°F)
- Air space: 16mm (5/8") with argon fill
- SHGC: 0.27-0.70 (depending on Low-E type)

**Triple-glazed systems:**
- U-value: 0.8-1.2 W/m²K (0.14-0.21 Btu/hr·ft²·°F)
- Two air spaces with argon or krypton fill
- SHGC: 0.25-0.50

### 2.2 METAL CLADDING SYSTEMS

#### Aluminum Curtain Wall
**Alloy 6063-T5:**
- Yield strength: 145 MPa (21,000 psi)
- Tensile strength: 186 MPa (27,000 psi)
- Thermal conductivity: 201 W/mK
- Thermal expansion: 23.6 × 10⁻⁶ /K
- Density: 2,700 kg/m³ (168 lb/ft³)

**Typical system components:**
- Mullion depths: 150mm, 200mm, 250mm (6", 8", 10")
- Glazing systems: Structural glazing, captured systems
- Thermal breaks: Polyamide strips 14-34mm wide
- Pressure equalization: Rainscreen principle

#### Steel Cladding Systems
**Pre-painted galvanized steel:**
- Base metal: G90 galvanized coating (275 g/m²)
- Paint system: PVDF or SMP
- Thickness: 0.5mm to 1.5mm (20 to 16 gauge)
- Expansion joint spacing: 30-40m (100-130')

### 2.3 MASONRY MATERIALS

#### Clay Brick Specifications
**Standard modular brick:**
- Dimensions: 194 × 92 × 57mm (7.625" × 3.625" × 2.25")
- Compressive strength: 20-100 MPa (3,000-14,500 psi)
- Water absorption: 8-22% (24-hour submersion)
- Thermal conductivity: 0.6-1.0 W/mK
- Density: 1,800-2,000 kg/m³ (112-125 lb/ft³)

**Face brick grades:**
- SW (Severe Weathering): Freeze-thaw resistance
- MW (Moderate Weathering): Moderate exposure
- NW (No Weathering): Interior use only

#### Concrete Masonry Units (CMU)
**Standard hollow block:**
- Dimensions: 390 × 190 × 190mm (15.375" × 7.625" × 7.625")
- Compressive strength: 8-35 MPa (1,200-5,000 psi)
- Density: 1,200-2,000 kg/m³ (75-125 lb/ft³)
- Thermal conductivity: 0.5-1.4 W/mK

**Specialty blocks:**
- Architectural split-face units
- Ground-face units
- Glazed units
- Interlocking units

## 3. INSULATION MATERIALS

### 3.1 THERMAL INSULATION PROPERTIES

#### Fiberglass Insulation
**Batt insulation:**
- R-value: 3.1-3.7 per inch (RSI 0.55-0.65 per 25mm)
- Density: 6-30 kg/m³ (0.4-1.9 lb/ft³)
- Operating temperature: -40°C to 230°C (-40°F to 450°F)
- Moisture resistance: Moderate (requires vapor barrier)

**Blown-in fiberglass:**
- R-value: 2.2-2.7 per inch (RSI 0.39-0.48 per 25mm)
- Settling factor: 20% over time
- Applications: Attics, wall cavities, irregular spaces

#### Polyurethane Foam Insulation
**Closed-cell spray foam:**
- R-value: 6.0-7.0 per inch (RSI 1.06-1.23 per 25mm)
- Density: 32-48 kg/m³ (2.0-3.0 lb/ft³)
- Vapor permeability: <1.0 perm
- Applications: Continuous insulation, air sealing

**Open-cell spray foam:**
- R-value: 3.5-4.0 per inch (RSI 0.62-0.70 per 25mm)
- Density: 6-8 kg/m³ (0.4-0.5 lb/ft³)
- Vapor permeability: 15-20 perms
- Applications: Wall cavities, sound dampening

#### Rigid Foam Insulation
**Extruded polystyrene (XPS):**
- R-value: 5.0 per inch (RSI 0.88 per 25mm)
- Compressive strength: 103-276 kPa (15-40 psi)
- Water absorption: <0.3%
- Applications: Foundation walls, under slabs

**Polyisocyanurate (Polyiso):**
- R-value: 6.5-7.0 per inch (RSI 1.15-1.23 per 25mm)
- Compressive strength: 138-276 kPa (20-40 psi)
- Fire resistance: Self-extinguishing
- Applications: Roof systems, wall sheathing

## 4. ENVIRONMENTAL PERFORMANCE DATA

### 4.1 Embodied Carbon Content (kgCO2e/kg)

**Concrete materials:**
- Portland cement: 0.820-0.940 kgCO2e/kg
- Ready-mix concrete (30 MPa): 0.130-0.150 kgCO2e/kg
- Reinforcing steel: 1.370-1.460 kgCO2e/kg

**Steel materials:**
- Structural steel (virgin): 1.850-2.100 kgCO2e/kg
- Structural steel (recycled): 0.450-0.650 kgCO2e/kg
- Stainless steel: 3.200-3.800 kgCO2e/kg

**Aluminum materials:**
- Primary aluminum: 8.200-9.200 kgCO2e/kg
- Recycled aluminum: 0.610-0.820 kgCO2e/kg

**Wood materials:**
- Softwood lumber: -0.900 to -0.700 kgCO2e/kg (carbon sequestration)
- Glulam beams: -0.650 to -0.450 kgCO2e/kg
- Cross-laminated timber: -0.580 to -0.380 kgCO2e/kg

### 4.2 Recycled Content Potential

**Concrete:**
- Fly ash replacement: Up to 30% of cement
- Recycled aggregate: Up to 100% for non-structural uses
- Recycled steel reinforcement: 90-95% recycled content

**Steel:**
- Structural steel: 90-95% recycled content typical
- Rebar: 95-98% recycled content typical

**Aluminum:**
- Extruded sections: 85-95% recycled content possible
- Sheet products: 70-85% recycled content typical

## 5. MATERIAL TESTING STANDARDS

### 5.1 ASTM Testing Standards

#### Concrete Testing
- ASTM C39: Compressive strength of concrete cylinders
- ASTM C78: Flexural strength of concrete
- ASTM C496: Splitting tensile strength of concrete
- ASTM C597: Pulse velocity through concrete
- ASTM C642: Density, absorption, and voids in concrete

#### Steel Testing
- ASTM A370: Standard test methods for steel products
- ASTM E8: Tension testing of metallic materials
- ASTM A6: General requirements for rolled steel plates
- ASTM A992: Structural steel for buildings

#### Thermal Performance Testing
- ASTM C177: Steady-state heat flux measurements
- ASTM C518: Steady-state thermal transmission properties
- ASTM E1131: Compositional analysis by thermogravimetry

### 5.2 International Testing Standards

#### ISO Standards
- ISO 6946: Building components thermal resistance calculation
- ISO 12570: Hygrothermal performance of building materials
- ISO 15148: Hygrothermal performance of building materials

#### EN European Standards
- EN 1992: Design of concrete structures (Eurocode 2)
- EN 1993: Design of steel structures (Eurocode 3)
- EN 1995: Design of timber structures (Eurocode 5)

This comprehensive materials database provides detailed specifications, performance characteristics, and testing standards essential for sophisticated architectural modeling and analysis systems."""

# Save the building materials database
with open('building_materials_database.txt', 'w') as f:
    f.write(building_materials_database)

print("Created building_materials_database.txt - Comprehensive material properties, specifications, and performance data")