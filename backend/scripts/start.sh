#!/bin/bash

# Production startup script for FastAPI backend on Render.com
# Optimized for performance, monitoring, and graceful shutdowns

set -e  # Exit on any error

# Default values
export PORT=${PORT:-10000}
export MAX_WORKERS=${MAX_WORKERS:-4}
export WORKER_TIMEOUT=${WORKER_TIMEOUT:-300}
export KEEP_ALIVE=${KEEP_ALIVE:-5}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

echo "Starting Text-to-CAD FastAPI Backend"
echo "Port: $PORT"
echo "Workers: $MAX_WORKERS"
echo "Timeout: $WORKER_TIMEOUT seconds"
echo "Keep Alive: $KEEP_ALIVE seconds"
echo "Log Level: $LOG_LEVEL"

# Pre-flight checks
echo "Running pre-flight checks..."

# Check if required environment variables are set
if [ -z "$DATABASE_URL" ]; then
    echo "ERROR: DATABASE_URL environment variable is not set"
    exit 1
fi

if [ -z "$REDIS_URL" ]; then
    echo "ERROR: REDIS_URL environment variable is not set"
    exit 1
fi

# Test database connectivity
echo "Testing database connectivity..."
python -c "
import asyncio
import sys
from app.database import test_connection

async def test_db():
    try:
        await test_connection()
        print('Database connection: OK')
    except Exception as e:
        print(f'Database connection failed: {e}')
        sys.exit(1)

asyncio.run(test_db())
" || exit 1

# Test Redis connectivity
echo "Testing Redis connectivity..."
python -c "
import redis
import sys
import os

try:
    r = redis.from_url(os.getenv('REDIS_URL'))
    r.ping()
    print('Redis connection: OK')
except Exception as e:
    print(f'Redis connection failed: {e}')
    sys.exit(1)
" || exit 1

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Initialize application data
echo "Initializing application data..."
python scripts/init_db.py

# Create log directory
mkdir -p /app/logs

# Calculate optimal number of workers based on CPU cores
WORKERS=${WEB_CONCURRENCY:-$MAX_WORKERS}

# Determine the best worker class
WORKER_CLASS="uvicorn.workers.UvicornWorker"

echo "Starting Gunicorn with $WORKERS workers..."

# Production startup with Gunicorn
exec gunicorn \
    app.main:app \
    --bind 0.0.0.0:$PORT \
    --workers $WORKERS \
    --worker-class $WORKER_CLASS \
    --worker-connections 1000 \
    --timeout $WORKER_TIMEOUT \
    --keep-alive $KEEP_ALIVE \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --preload \
    --log-level $LOG_LEVEL \
    --access-logfile /app/logs/access.log \
    --error-logfile /app/logs/error.log \
    --capture-output \
    --enable-stdio-inheritance