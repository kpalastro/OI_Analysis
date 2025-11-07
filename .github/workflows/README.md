# CI/CD Pipeline

This directory contains GitHub Actions workflows for automated testing and deployment.

## Workflows

### deploy.yml
Automated CI/CD pipeline that:
1. **Tests**: Runs linting and syntax checks on every push
2. **Deploys**: Automatically deploys to production server when changes are pushed to `main` branch

## Setup Instructions

### 1. Configure GitHub Secrets (REQUIRED)

**⚠️ IMPORTANT: Deployment will fail if secrets are not set!**

Go to your repository → **Settings → Secrets and variables → Actions → New repository secret**

Add the following 4 secrets:

#### Secret 1: `HOST_IP`
- **Value**: `aiagents4biz.in` (or your server IP address)
- **Example**: `aiagents4biz.in` or `192.168.1.100`

#### Secret 2: `SSH_USER`
- **Value**: Your SSH username on the server
- **Example**: `ubuntu`, `root`, or `www-data`

#### Secret 3: `SSH_PRIVATE_KEY`
- **Value**: Your SSH private key content (entire key including `-----BEGIN` and `-----END` lines)
- **How to generate**:
  ```bash
  # On your local machine
  ssh-keygen -t ed25519 -C "github-actions@aiagents4biz.in" -f ~/.ssh/github_actions_deploy
  
  # Display private key (copy entire output including BEGIN/END lines)
  cat ~/.ssh/github_actions_deploy
  
  # Add public key to server
  ssh-copy-id -i ~/.ssh/github_actions_deploy.pub user@aiagents4biz.in
  ```
- **Important**: Copy the ENTIRE key including:
  ```
  -----BEGIN OPENSSH PRIVATE KEY-----
  [key content]
  -----END OPENSSH PRIVATE KEY-----
  ```

#### Secret 4: `DEPLOY_PATH`
- **Value**: `/var/www/oi-tracker`
- **Example**: `/var/www/oi-tracker`

### 2. Verify Secrets Are Set

After adding secrets, you can verify by:
1. Going to **Settings → Secrets and variables → Actions**
2. You should see all 4 secrets listed
3. The workflow will now validate them before deployment

### 2. Generate SSH Key for GitHub Actions

On your local machine:
```bash
ssh-keygen -t rsa -b 4096 -C "github-actions@aiagents4biz.in" -f ~/.ssh/github_actions_deploy
```

Add public key to server:
```bash
ssh-copy-id -i ~/.ssh/github_actions_deploy.pub user@aiagents4biz.in
```

Add private key to GitHub Secrets:
```bash
cat ~/.ssh/github_actions_deploy
# Copy the output and paste it into GitHub Secrets as SSH_PRIVATE_KEY
```

### 3. Verify Deployment

After pushing to `main` branch:
1. Check GitHub Actions tab for workflow status
2. Verify deployment on server
3. Test the application at http://aiagents4biz.in/login

## Manual Deployment

If you need to deploy manually, see `deploy/deploy.sh` and `deploy/README.md`

