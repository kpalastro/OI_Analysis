# CI/CD Pipeline

This directory contains GitHub Actions workflows for automated testing and deployment.

## Workflows

### deploy.yml
Automated CI/CD pipeline that:
1. **Tests**: Runs linting and syntax checks on every push
2. **Deploys**: Automatically deploys to production server when changes are pushed to `main` branch

## Setup Instructions

### 1. Configure GitHub Secrets

Go to your repository → Settings → Secrets and variables → Actions

Add the following secrets:

- **SSH_PRIVATE_KEY**: Private SSH key for server access
- **HOST_IP**: Server IP or domain (aiagents4biz.in)
- **SSH_USER**: SSH username
- **DEPLOY_PATH**: Deployment path (/var/www/oi-tracker)

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

