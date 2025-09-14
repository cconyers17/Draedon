"""
Health check endpoints for monitoring and deployment
Provides comprehensive health status for the Text-to-CAD backend
"""

import time
import psutil
import asyncio
from typing import Dict, Any
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import text
from redis import Redis

from ..database import get_database
from ..config import get_settings
from ..core.cache import get_redis_client

router = APIRouter()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check endpoint for monitoring
    Returns detailed status of all system components
    """
    start_time = time.time()

    try:
        # Basic system information
        health_status = {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "text-to-cad-backend",
            "version": "1.0.0",
            "environment": get_settings().environment,
            "uptime": time.time() - psutil.boot_time(),
        }

        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        health_status["system"] = {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_mb": round(memory.available / 1024 / 1024, 2),
            "disk_usage_percent": disk.percent,
            "disk_free_gb": round(disk.free / 1024 / 1024 / 1024, 2),
        }

        # Check database connectivity
        db_status = await check_database_health()
        health_status["database"] = db_status

        # Check Redis connectivity
        redis_status = await check_redis_health()
        health_status["redis"] = redis_status

        # Check external APIs
        api_status = await check_external_apis()
        health_status["external_apis"] = api_status

        # Performance metrics
        response_time = round((time.time() - start_time) * 1000, 2)
        health_status["performance"] = {
            "response_time_ms": response_time,
            "status": "ok" if response_time < 1000 else "slow"
        }

        # Overall health determination
        components_healthy = all([
            db_status["status"] == "connected",
            redis_status["status"] == "connected",
            cpu_percent < 90,
            memory.percent < 90,
            disk.percent < 90
        ])

        health_status["status"] = "ok" if components_healthy else "degraded"

        return health_status

    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "response_time_ms": round((time.time() - start_time) * 1000, 2)
        }

@router.get("/health/ready")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness probe for Kubernetes/container orchestration
    Checks if service is ready to accept traffic
    """
    try:
        # Quick checks for readiness
        db_status = await check_database_health()
        redis_status = await check_redis_health()

        ready = (
            db_status["status"] == "connected" and
            redis_status["status"] == "connected"
        )

        return {
            "ready": ready,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {
                "database": db_status["status"] == "connected",
                "redis": redis_status["status"] == "connected"
            }
        }

    except Exception as e:
        return {
            "ready": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }

@router.get("/health/live")
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness probe for Kubernetes/container orchestration
    Simple check that service is alive
    """
    return {
        "alive": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "text-to-cad-backend"
    }

async def check_database_health() -> Dict[str, Any]:
    """Check database connectivity and performance"""
    try:
        start_time = time.time()

        # Get database connection
        db = await get_database()

        # Execute simple query
        result = await db.execute(text("SELECT 1"))
        await result.fetchone()

        response_time = round((time.time() - start_time) * 1000, 2)

        return {
            "status": "connected",
            "response_time_ms": response_time,
            "type": "postgresql"
        }

    except Exception as e:
        return {
            "status": "disconnected",
            "error": str(e),
            "type": "postgresql"
        }

async def check_redis_health() -> Dict[str, Any]:
    """Check Redis connectivity and performance"""
    try:
        start_time = time.time()

        # Get Redis client
        redis_client = await get_redis_client()

        # Ping Redis
        await redis_client.ping()

        response_time = round((time.time() - start_time) * 1000, 2)

        return {
            "status": "connected",
            "response_time_ms": response_time,
            "type": "redis"
        }

    except Exception as e:
        return {
            "status": "disconnected",
            "error": str(e),
            "type": "redis"
        }

async def check_external_apis() -> Dict[str, Any]:
    """Check external API availability"""
    settings = get_settings()

    api_status = {
        "meshy_ai": {
            "configured": bool(getattr(settings, 'meshy_ai_api_key', None)),
            "status": "unknown"
        },
        "trellis_3d": {
            "configured": bool(getattr(settings, 'trellis_3d_api_key', None)),
            "status": "unknown"
        },
        "rodin_ai": {
            "configured": bool(getattr(settings, 'rodin_ai_api_key', None)),
            "status": "unknown"
        }
    }

    # You could add actual API ping tests here if needed
    # For now, just report configuration status

    return api_status

@router.get("/metrics")
async def metrics_endpoint() -> Dict[str, Any]:
    """
    Prometheus-style metrics endpoint
    Returns detailed metrics for monitoring systems
    """
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()

        return {
            "text_to_cad_cpu_usage_percent": cpu_percent,
            "text_to_cad_memory_usage_percent": memory.percent,
            "text_to_cad_memory_available_bytes": memory.available,
            "text_to_cad_disk_usage_percent": disk.percent,
            "text_to_cad_disk_free_bytes": disk.free,
            "text_to_cad_process_memory_rss_bytes": process_memory.rss,
            "text_to_cad_process_memory_vms_bytes": process_memory.vms,
            "text_to_cad_uptime_seconds": time.time() - psutil.boot_time(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")