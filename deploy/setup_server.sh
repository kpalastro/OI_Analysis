#!/bin/bash
# Server Setup Script for OI Tracker Application
# Run this script on your Ubuntu server (aiagents4biz.in)

set -e

echo "=========================================="
echo "OI Tracker Server Setup Script"
echo "=========================================="

# Configuration
DEPLOY_USER="${DEPLOY_USER:-$USER}"
DEPLOY_PATH="${DEPLOY_PATH:-/var/www/oi-tracker}"
APP_NAME="oi-tracker"
PYTHON_VERSION="3.12"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root or with sudo${NC}"
    exit 1
fi

echo -e "${GREEN}Step 1: Installing system dependencies...${NC}"
apt-get update
apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    nginx \
    supervisor \
    sqlite3 \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev

echo -e "${GREEN}Step 2: Creating deployment directory...${NC}"
mkdir -p $DEPLOY_PATH
chown $DEPLOY_USER:$DEPLOY_USER $DEPLOY_PATH

echo -e "${GREEN}Step 3: Setting up Python virtual environment...${NC}"
cd $DEPLOY_PATH
if [ ! -d "venv" ]; then
    sudo -u $DEPLOY_USER python3 -m venv venv
fi

echo -e "${GREEN}Step 4: Creating application directories...${NC}"
sudo -u $DEPLOY_USER mkdir -p logs
sudo -u $DEPLOY_USER mkdir -p ml_system/models
sudo -u $DEPLOY_USER mkdir -p ml_system/training/reports

echo -e "${GREEN}Step 5: Setting up systemd service...${NC}"
cat > /etc/systemd/system/${APP_NAME}.service << EOF
[Unit]
Description=OI Tracker Web Application
After=network.target

[Service]
Type=simple
User=$DEPLOY_USER
Group=$DEPLOY_USER
WorkingDirectory=$DEPLOY_PATH
Environment="PATH=$DEPLOY_PATH/venv/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=$DEPLOY_PATH/venv/bin/python $DEPLOY_PATH/oi_tracker_web.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=oi-tracker

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ${APP_NAME}

echo -e "${GREEN}Step 6: Setting up Nginx reverse proxy...${NC}"
cat > /etc/nginx/sites-available/${APP_NAME} << EOF
server {
    listen 80;
    server_name aiagents4biz.in www.aiagents4biz.in;

    client_max_body_size 10M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Static files (if needed)
    location /static {
        alias $DEPLOY_PATH/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
EOF

# Enable site
ln -sf /etc/nginx/sites-available/${APP_NAME} /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

echo -e "${GREEN}Step 7: Setting up SSL with Let's Encrypt (optional)...${NC}"
echo -e "${YELLOW}To enable SSL, run:${NC}"
echo "  sudo apt-get install certbot python3-certbot-nginx"
echo "  sudo certbot --nginx -d aiagents4biz.in -d www.aiagents4biz.in"

echo -e "${GREEN}Step 8: Setting up firewall...${NC}"
if command -v ufw &> /dev/null; then
    ufw allow 22/tcp
    ufw allow 80/tcp
    ufw allow 443/tcp
    ufw --force enable
    echo -e "${GREEN}Firewall configured${NC}"
fi

echo -e "${GREEN}=========================================="
echo "Server setup completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Clone your repository to $DEPLOY_PATH"
echo "2. Create .env file with your credentials"
echo "3. Install Python dependencies: cd $DEPLOY_PATH && source venv/bin/activate && pip install -r requirements.txt"
echo "4. Run database migration: python3 database_migration.py"
echo "5. Start the service: sudo systemctl start ${APP_NAME}"
echo "6. Check status: sudo systemctl status ${APP_NAME}"
echo "7. View logs: sudo journalctl -u ${APP_NAME} -f"
echo ""
echo -e "${YELLOW}Note: Make sure to configure your .env file before starting!${NC}"

