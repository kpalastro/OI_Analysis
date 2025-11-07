# Deployment Guide for aiagents4biz.in

This guide will help you deploy the OI Tracker application to your Ubuntu server.

## Prerequisites

- Ubuntu 20.04 or later
- Root or sudo access
- Domain name: aiagents4biz.in
- Git repository access

## Step 1: Initial Server Setup

### 1.1 Connect to your server
```bash
ssh user@aiagents4biz.in
```

### 1.2 Run the setup script
```bash
# Clone the repository first (or upload the setup script)
git clone https://github.com/your-username/OI_Analysis.git /var/www/oi-tracker

# Make setup script executable
chmod +x deploy/setup_server.sh

# Run setup (as root or with sudo)
sudo ./deploy/setup_server.sh
```

This script will:
- Install system dependencies (Python, Nginx, etc.)
- Create deployment directory
- Set up Python virtual environment
- Configure systemd service
- Set up Nginx reverse proxy
- Configure firewall

## Step 2: Configure Environment Variables

### 2.1 Create .env file
```bash
cd /var/www/oi-tracker
nano .env
```

Add your credentials:
```env
# Zerodha Credentials
ZERODHA_USER_ID=your_user_id
ZERODHA_PASSWORD=your_password

# Flask Configuration
FLASK_SECRET_KEY=your-secret-key-change-this
FLASK_HOST=127.0.0.1
FLASK_PORT=5000

# Application Configuration
REFRESH_INTERVAL_SECONDS=30
OPTIONS_COUNT=5
STRIKE_DIFFERENCE=50
```

### 2.2 Secure the .env file
```bash
chmod 600 .env
chown www-data:www-data .env
```

## Step 3: Install Dependencies

```bash
cd /var/www/oi-tracker
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r ml_system/requirements_ml.txt
```

## Step 4: Initialize Database

```bash
cd /var/www/oi-tracker
source venv/bin/activate
python3 database_migration.py
python3 -c "import database; database.initialize_database()"
```

## Step 5: Start the Service

```bash
sudo systemctl start oi-tracker
sudo systemctl enable oi-tracker
sudo systemctl status oi-tracker
```

## Step 6: Configure SSL (Optional but Recommended)

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d aiagents4biz.in -d www.aiagents4biz.in

# Auto-renewal is set up automatically
```

## Step 7: Configure GitHub Actions Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add the following secrets:

1. **SSH_PRIVATE_KEY**: Your server's SSH private key
   ```bash
   # Generate SSH key pair if needed
   ssh-keygen -t rsa -b 4096 -C "github-actions@aiagents4biz.in"
   
   # Copy private key to GitHub Secrets
   cat ~/.ssh/id_rsa
   
   # Add public key to server
   ssh-copy-id user@aiagents4biz.in
   ```

2. **HOST_IP**: Your server IP address or domain
   ```
   aiagents4biz.in
   ```

3. **SSH_USER**: SSH username
   ```
   your_username
   ```

4. **DEPLOY_PATH**: Deployment path
   ```
   /var/www/oi-tracker
   ```

## Step 8: Verify Deployment

### Check service status
```bash
sudo systemctl status oi-tracker
```

### View logs
```bash
sudo journalctl -u oi-tracker -f
```

### Test the application
```bash
curl http://localhost:5000/login
```

### Access via domain
Open your browser and visit:
```
http://aiagents4biz.in/login
```

## Manual Deployment

If you need to deploy manually (without GitHub Actions):

```bash
cd /var/www/oi-tracker
chmod +x deploy/deploy.sh
./deploy/deploy.sh
```

## Troubleshooting

### Service won't start
```bash
# Check logs
sudo journalctl -u oi-tracker -n 50

# Check if port is in use
sudo netstat -tlnp | grep 5000

# Check permissions
ls -la /var/www/oi-tracker
```

### Database errors
```bash
# Check database file
ls -la oi_tracker.db

# Run migration again
python3 database_migration.py
```

### Nginx errors
```bash
# Test configuration
sudo nginx -t

# Check Nginx logs
sudo tail -f /var/log/nginx/error.log
```

### Permission issues
```bash
# Fix ownership
sudo chown -R www-data:www-data /var/www/oi-tracker
sudo chmod -R 755 /var/www/oi-tracker
```

## Monitoring

### View application logs
```bash
sudo journalctl -u oi-tracker -f
```

### View Nginx logs
```bash
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Check system resources
```bash
# CPU and Memory
htop

# Disk space
df -h

# Application process
ps aux | grep oi_tracker_web
```

## Maintenance

### Update application
```bash
cd /var/www/oi-tracker
git pull origin main
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart oi-tracker
```

### Backup database
```bash
# Create backup
cp oi_tracker.db backups/oi_tracker_$(date +%Y%m%d_%H%M%S).db

# Or use cron for automated backups
```

### Restart service
```bash
sudo systemctl restart oi-tracker
```

## Security Considerations

1. **Firewall**: Ensure only necessary ports are open (22, 80, 443)
2. **SSL**: Always use HTTPS in production
3. **Environment Variables**: Never commit .env file to Git
4. **SSH Keys**: Use SSH keys instead of passwords
5. **Updates**: Keep system packages updated
6. **Backups**: Regularly backup your database

## Support

For issues or questions, check:
- Application logs: `sudo journalctl -u oi-tracker -f`
- Nginx logs: `/var/log/nginx/error.log`
- GitHub Issues: Your repository issues page

