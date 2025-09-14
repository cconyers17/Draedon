"""
Redis caching utilities for performance optimization.
"""

import json
import hashlib
from typing import Optional, Any, Union
import redis.asyncio as redis
from redis.asyncio import ConnectionPool
import structlog

from app.core.config import settings

logger = structlog.get_logger()

# Redis connection pool
redis_pool: Optional[ConnectionPool] = None
redis_client: Optional[redis.Redis] = None


async def init_redis():
    """
    Initialize Redis connection pool.
    """
    global redis_pool, redis_client

    redis_pool = ConnectionPool.from_url(
        str(settings.REDIS_URL),
        max_connections=50,
        decode_responses=True
    )
    redis_client = redis.Redis(connection_pool=redis_pool)

    # Test connection
    try:
        await redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error("Failed to connect to Redis", error=str(e))
        raise


async def close_redis():
    """
    Close Redis connection pool.
    """
    global redis_pool, redis_client

    if redis_client:
        await redis_client.close()

    if redis_pool:
        await redis_pool.disconnect()

    logger.info("Redis connection closed")


def cache_key_wrapper(*args) -> str:
    """
    Generate cache key from arguments.
    """
    key_parts = []
    for arg in args:
        if isinstance(arg, (dict, list)):
            key_parts.append(hashlib.md5(json.dumps(arg, sort_keys=True).encode()).hexdigest())
        else:
            key_parts.append(str(arg))

    return ":".join(key_parts)


async def get_cache(key: str) -> Optional[Any]:
    """
    Get value from cache.
    """
    if not redis_client:
        return None

    try:
        value = await redis_client.get(key)
        if value:
            return json.loads(value)
        return None
    except Exception as e:
        logger.warning("Cache get failed", key=key, error=str(e))
        return None


async def set_cache(key: str, value: Any, ttl: int = 3600) -> bool:
    """
    Set value in cache with TTL.
    """
    if not redis_client:
        return False

    try:
        serialized = json.dumps(value)
        await redis_client.setex(key, ttl, serialized)
        return True
    except Exception as e:
        logger.warning("Cache set failed", key=key, error=str(e))
        return False


async def delete_cache(key: str) -> bool:
    """
    Delete key from cache.
    """
    if not redis_client:
        return False

    try:
        await redis_client.delete(key)
        return True
    except Exception as e:
        logger.warning("Cache delete failed", key=key, error=str(e))
        return False


async def invalidate_pattern(pattern: str) -> int:
    """
    Invalidate all keys matching pattern.
    """
    if not redis_client:
        return 0

    try:
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = await redis_client.scan(cursor, match=pattern, count=100)
            if keys:
                deleted += await redis_client.delete(*keys)
            if cursor == 0:
                break
        return deleted
    except Exception as e:
        logger.warning("Pattern invalidation failed", pattern=pattern, error=str(e))
        return 0


class CacheDecorator:
    """
    Decorator for caching function results.
    """

    def __init__(self, ttl: int = 3600, key_prefix: str = ""):
        self.ttl = ttl
        self.key_prefix = key_prefix

    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_key_wrapper(
                self.key_prefix,
                func.__name__,
                *args,
                **kwargs
            )

            # Try to get from cache
            cached = await get_cache(cache_key)
            if cached is not None:
                logger.debug("Cache hit", function=func.__name__, key=cache_key)
                return cached

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            await set_cache(cache_key, result, self.ttl)
            logger.debug("Cache miss, stored", function=func.__name__, key=cache_key)

            return result

        return wrapper


# Usage example:
# @CacheDecorator(ttl=3600, key_prefix="nlp")
# async def process_text(text: str):
#     # Processing logic
#     return result