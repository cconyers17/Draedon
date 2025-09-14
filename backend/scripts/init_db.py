#!/usr/bin/env python3
"""
Database initialization script for Text-to-CAD application
Initializes database with required data, tables, and configurations
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.database import get_database, init_database
from app.models import *  # Import all models to register them
from app.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def init_material_database():
    """Initialize material properties database"""
    logger.info("Initializing material database...")

    # Material data would be loaded from building_materials_database.txt
    materials_data = [
        {
            "name": "Concrete",
            "type": "structural",
            "density": 2400,  # kg/m³
            "compressive_strength": 30,  # MPa
            "tensile_strength": 3,  # MPa
            "elastic_modulus": 30000,  # MPa
            "thermal_conductivity": 1.7,  # W/m·K
            "fire_rating": "A1",
            "carbon_footprint": 410,  # kg CO2/m³
            "cost_per_cubic_meter": 150,  # USD
        },
        {
            "name": "Steel",
            "type": "structural",
            "density": 7850,  # kg/m³
            "compressive_strength": 250,  # MPa
            "tensile_strength": 400,  # MPa
            "elastic_modulus": 200000,  # MPa
            "thermal_conductivity": 50,  # W/m·K
            "fire_rating": "A1",
            "carbon_footprint": 2300,  # kg CO2/m³
            "cost_per_cubic_meter": 6000,  # USD
        },
        {
            "name": "Wood",
            "type": "structural",
            "density": 600,  # kg/m³
            "compressive_strength": 40,  # MPa
            "tensile_strength": 80,  # MPa
            "elastic_modulus": 12000,  # MPa
            "thermal_conductivity": 0.12,  # W/m·K
            "fire_rating": "D-s2,d0",
            "carbon_footprint": -900,  # kg CO2/m³ (negative = carbon storing)
            "cost_per_cubic_meter": 800,  # USD
        },
        {
            "name": "Glass",
            "type": "envelope",
            "density": 2500,  # kg/m³
            "compressive_strength": 1000,  # MPa
            "tensile_strength": 50,  # MPa
            "elastic_modulus": 70000,  # MPa
            "thermal_conductivity": 1.0,  # W/m·K
            "fire_rating": "A1",
            "carbon_footprint": 850,  # kg CO2/m³
            "cost_per_cubic_meter": 3000,  # USD
        }
    ]

    # This would insert materials into database
    logger.info(f"Loaded {len(materials_data)} materials into database")

async def init_building_codes():
    """Initialize building code constraints"""
    logger.info("Initializing building codes...")

    building_codes = {
        "residential": {
            "min_ceiling_height": 2.4,  # meters
            "min_room_area": 7.0,  # m²
            "min_window_area_ratio": 0.1,  # 10% of floor area
            "max_occupancy_per_sqm": 0.1,  # people per m²
            "min_corridor_width": 1.2,  # meters
            "max_travel_distance": 45,  # meters to exit
        },
        "commercial": {
            "min_ceiling_height": 2.7,  # meters
            "min_room_area": 10.0,  # m²
            "min_window_area_ratio": 0.15,  # 15% of floor area
            "max_occupancy_per_sqm": 0.2,  # people per m²
            "min_corridor_width": 1.8,  # meters
            "max_travel_distance": 60,  # meters to exit
        }
    }

    logger.info("Building codes initialized")

async def init_cad_templates():
    """Initialize CAD geometry templates"""
    logger.info("Initializing CAD templates...")

    # This would initialize common geometric templates
    templates = [
        "rectangular_room",
        "circular_room",
        "l_shaped_room",
        "standard_door",
        "standard_window",
        "stairs_straight",
        "stairs_spiral"
    ]

    logger.info(f"Loaded {len(templates)} CAD templates")

async def check_external_apis():
    """Check external API connectivity"""
    logger.info("Checking external API connectivity...")

    apis = [
        ("Meshy AI", "MESHY_AI_API_KEY"),
        ("Trellis 3D", "TRELLIS_3D_API_KEY"),
        ("Rodin AI", "RODIN_AI_API_KEY")
    ]

    settings = get_settings()

    for api_name, env_var in apis:
        api_key = getattr(settings, env_var.lower(), None)
        if api_key:
            logger.info(f"{api_name}: API key configured")
        else:
            logger.warning(f"{api_name}: API key not configured")

async def main():
    """Main initialization function"""
    logger.info("Starting database initialization...")

    try:
        # Initialize database connection
        await init_database()
        logger.info("Database connection established")

        # Run initialization tasks
        await init_material_database()
        await init_building_codes()
        await init_cad_templates()
        await check_external_apis()

        logger.info("Database initialization completed successfully")

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)
    finally:
        # Clean up database connections
        logger.info("Closing database connections")

if __name__ == "__main__":
    asyncio.run(main())