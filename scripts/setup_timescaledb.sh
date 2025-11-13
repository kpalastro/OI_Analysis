#!/bin/bash
# Setup TimescaleDB in PostgreSQL Docker container
# Usage: ./setup_timescaledb.sh <container_name>

CONTAINER_NAME=${1:-"your_postgres_container"}

echo "Setting up TimescaleDB in container: $CONTAINER_NAME"

# Check if container exists
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Container '$CONTAINER_NAME' not found"
    echo "Available containers:"
    docker ps -a --format '{{.Names}}'
    exit 1
fi

# Method 1: If using TimescaleDB official image, just add to postgresql.conf
echo "Configuring shared_preload_libraries..."

# Find postgresql.conf location
PG_CONF=$(docker exec $CONTAINER_NAME psql -U root -d postgres -t -c "SHOW config_file;" | xargs)

if [ -z "$PG_CONF" ]; then
    echo "Could not find postgresql.conf. Trying default locations..."
    # Try common locations
    for loc in "/var/lib/postgresql/data/postgresql.conf" "/etc/postgresql/postgresql.conf"; do
        if docker exec $CONTAINER_NAME test -f "$loc"; then
            PG_CONF="$loc"
            break
        fi
    done
fi

if [ -z "$PG_CONF" ]; then
    echo "Error: Could not locate postgresql.conf"
    echo "Please manually edit postgresql.conf and add:"
    echo "  shared_preload_libraries = 'timescaledb'"
    exit 1
fi

echo "Found postgresql.conf at: $PG_CONF"

# Backup the config
docker exec $CONTAINER_NAME cp "$PG_CONF" "${PG_CONF}.backup"

# Check if timescaledb is already in shared_preload_libraries
if docker exec $CONTAINER_NAME grep -q "shared_preload_libraries.*timescaledb" "$PG_CONF"; then
    echo "TimescaleDB already configured in shared_preload_libraries"
else
    # Add or modify shared_preload_libraries
    if docker exec $CONTAINER_NAME grep -q "^shared_preload_libraries" "$PG_CONF"; then
        # Modify existing line
        docker exec $CONTAINER_NAME sed -i "s/^shared_preload_libraries.*/shared_preload_libraries = 'timescaledb'/" "$PG_CONF"
    else
        # Add new line
        docker exec $CONTAINER_NAME sh -c "echo \"shared_preload_libraries = 'timescaledb'\" >> $PG_CONF"
    fi
    echo "Added timescaledb to shared_preload_libraries"
fi

echo ""
echo "✅ Configuration updated. You need to restart the container for changes to take effect:"
echo "   docker restart $CONTAINER_NAME"
echo ""
echo "After restart, create the extension:"
echo "   docker exec $CONTAINER_NAME psql -U root -d oi_tracker -c \"CREATE EXTENSION IF NOT EXISTS timescaledb;\""

