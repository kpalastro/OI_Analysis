#!/bin/bash
# Manual Deployment Script
# Run this script on the server to deploy the application

set -e

# Configuration
DEPLOY_PATH="${DEPLOY_PATH:-/var/www/oi-tracker}"
APP_NAME="oi-tracker"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting deployment...${NC}"

cd $DEPLOY_PATH || exit 1

# Activate virtual environment
source venv/bin/activate

# Pull latest changes
echo -e "${GREEN}Pulling latest changes...${NC}"
git fetch origin
git reset --hard origin/main

# Install/update dependencies
echo -e "${GREEN}Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
if [ -f ml_system/requirements_ml.txt ]; then
    pip install -r ml_system/requirements_ml.txt
fi

# Run database migration
if [ -f database_migration.py ]; then
    echo -e "${GREEN}Running database migration...${NC}"
    python3 database_migration.py || true
fi

# Initialize database if needed
if [ ! -f oi_tracker.db ]; then
    echo -e "${YELLOW}Initializing database...${NC}"
    python3 -c "import database; database.initialize_database()" || true
fi

# Restart service
echo -e "${GREEN}Restarting service...${NC}"
sudo systemctl restart $APP_NAME

# Wait a moment
sleep 3

# Check status
echo -e "${GREEN}Checking service status...${NC}"
sudo systemctl status $APP_NAME --no-pager || true

echo -e "${GREEN}Deployment completed!${NC}"
echo ""
echo "View logs: sudo journalctl -u $APP_NAME -f"

