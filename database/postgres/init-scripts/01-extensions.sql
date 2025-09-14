-- Initialize PostgreSQL extensions for Text-to-CAD application
-- This script runs during database initialization

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "hstore";

-- PostGIS for spatial data (if available)
CREATE EXTENSION IF NOT EXISTS "postgis";
CREATE EXTENSION IF NOT EXISTS "postgis_topology";

-- Full-text search
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "unaccent";

-- JSON operations
CREATE EXTENSION IF NOT EXISTS "jsonb_plpython3u" IF EXISTS;

-- Performance extensions
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create custom functions for CAD operations
CREATE OR REPLACE FUNCTION calculate_bounding_box(geometry_data JSONB)
RETURNS JSONB AS $$
BEGIN
    -- Calculate 3D bounding box from geometry data
    -- This is a placeholder for actual geometric calculations
    RETURN jsonb_build_object(
        'min_x', (geometry_data->>'min_x')::NUMERIC,
        'min_y', (geometry_data->>'min_y')::NUMERIC,
        'min_z', (geometry_data->>'min_z')::NUMERIC,
        'max_x', (geometry_data->>'max_x')::NUMERIC,
        'max_y', (geometry_data->>'max_y')::NUMERIC,
        'max_z', (geometry_data->>'max_z')::NUMERIC
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create function to validate material properties
CREATE OR REPLACE FUNCTION validate_material_properties(properties JSONB)
RETURNS BOOLEAN AS $$
BEGIN
    -- Validate that required material properties are present
    RETURN (
        properties ? 'density' AND
        properties ? 'compressive_strength' AND
        properties ? 'thermal_conductivity' AND
        (properties->>'density')::NUMERIC > 0 AND
        (properties->>'compressive_strength')::NUMERIC > 0 AND
        (properties->>'thermal_conductivity')::NUMERIC > 0
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Create function to calculate material cost
CREATE OR REPLACE FUNCTION calculate_material_cost(
    material_id UUID,
    volume NUMERIC
) RETURNS NUMERIC AS $$
DECLARE
    cost_per_unit NUMERIC;
BEGIN
    -- Get cost per unit from materials table
    SELECT cost_per_cubic_meter INTO cost_per_unit
    FROM materials
    WHERE id = material_id;

    -- Return total cost
    RETURN COALESCE(cost_per_unit * volume, 0);
END;
$$ LANGUAGE plpgsql STABLE;

-- Create indexes for performance
-- Note: Table-specific indexes will be created by Alembic migrations

-- Enable row-level security
ALTER DATABASE textcad_production SET row_security = on;