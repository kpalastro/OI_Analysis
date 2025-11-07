#!/bin/bash
# Database Backup Script
# Run this via cron for automated backups

set -e

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/var/backups/oi-tracker}"
DEPLOY_PATH="${DEPLOY_PATH:-/var/www/oi-tracker}"
RETENTION_DAYS=30

# Create backup directory
mkdir -p $BACKUP_DIR

# Generate backup filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/oi_tracker_$TIMESTAMP.db"

# Backup database
if [ -f "$DEPLOY_PATH/oi_tracker.db" ]; then
    cp "$DEPLOY_PATH/oi_tracker.db" "$BACKUP_FILE"
    gzip "$BACKUP_FILE"
    echo "Backup created: ${BACKUP_FILE}.gz"
else
    echo "Database file not found: $DEPLOY_PATH/oi_tracker.db"
    exit 1
fi

# Remove old backups
find $BACKUP_DIR -name "oi_tracker_*.db.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup completed successfully"

