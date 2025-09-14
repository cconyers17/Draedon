# PostgreSQL Dockerfile optimized for Text-to-CAD application
# Includes spatial extensions and optimized configuration

FROM postgres:15-alpine

# Install PostGIS and other extensions
RUN apk add --no-cache \
    postgis \
    postgis-dev \
    postgresql-contrib

# Set environment variables
ENV POSTGRES_DB=textcad_production
ENV POSTGRES_USER=textcad_user
ENV POSTGRES_INITDB_ARGS="--encoding=UTF8 --locale=C"

# Copy initialization scripts
COPY ./init-scripts/ /docker-entrypoint-initdb.d/

# Copy optimized PostgreSQL configuration
COPY ./postgresql.conf /usr/local/share/postgresql/postgresql.conf.sample

# Create data directory
RUN mkdir -p /var/lib/postgresql/data && \
    chown -R postgres:postgres /var/lib/postgresql/data

# Expose PostgreSQL port
EXPOSE 5432

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD pg_isready -U $POSTGRES_USER -d $POSTGRES_DB || exit 1

# Use default entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["postgres", "-c", "config_file=/usr/local/share/postgresql/postgresql.conf.sample"]