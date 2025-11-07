# Quick Start: Deployment to aiagents4biz.in

## 🚀 Quick Setup (5 minutes)

### 1. On Your Server (aiagents4biz.in)

```bash
# Connect to server
ssh user@aiagents4biz.in

# Clone repository
sudo git clone https://github.com/your-username/OI_Analysis.git /var/www/oi-tracker

# Run setup script
cd /var/www/oi-tracker
sudo chmod +x deploy/setup_server.sh
sudo ./deploy/setup_server.sh

# Create .env file
sudo nano /var/www/oi-tracker/.env
# (Add your credentials - see deploy/.env.example)

# Install dependencies
cd /var/www/oi-tracker
source venv/bin/activate
pip install -r requirements.txt
pip install -r ml_system/requirements_ml.txt

# Initialize database
python3 database_migration.py

# Start service
sudo systemctl start oi-tracker
sudo systemctl status oi-tracker
```

### 2. Configure GitHub Secrets

Go to: **GitHub Repository → Settings → Secrets and variables → Actions**

Add these 4 secrets:

1. **SSH_PRIVATE_KEY**
   ```bash
   # Generate key pair
   ssh-keygen -t rsa -b 4096 -C "github-actions@aiagents4biz.in" -f ~/.ssh/github_actions
   
   # Add public key to server
   ssh-copy-id -i ~/.ssh/github_actions.pub user@aiagents4biz.in
   
   # Copy private key (for GitHub Secret)
   cat ~/.ssh/github_actions
   ```

2. **HOST_IP**: `aiagents4biz.in`

3. **SSH_USER**: Your SSH username (e.g., `ubuntu`, `root`)

4. **DEPLOY_PATH**: `/var/www/oi-tracker`

### 3. Enable SSL (Optional)

```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d aiagents4biz.in -d www.aiagents4biz.in
```

### 4. Test Deployment

Push to main branch:
```bash
git push origin main
```

Check GitHub Actions tab - deployment should start automatically!

## 📋 What Gets Deployed

- ✅ Automatic testing before deployment
- ✅ Zero-downtime deployment
- ✅ Database migration
- ✅ Service auto-restart
- ✅ Health check verification

## 📚 Full Documentation

See `deploy/README.md` for complete setup instructions.

## 🔧 Troubleshooting

**Service won't start:**
```bash
sudo journalctl -u oi-tracker -f
```

**Deployment fails:**
- Check GitHub Actions logs
- Verify SSH keys are correct
- Ensure .env file exists on server

**Application not accessible:**
```bash
# Check service
sudo systemctl status oi-tracker

# Check Nginx
sudo nginx -t
sudo systemctl status nginx

# Check logs
sudo tail -f /var/log/nginx/error.log
```

