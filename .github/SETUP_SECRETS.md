# Setting Up GitHub Secrets for CI/CD

## Quick Setup Guide

### Step 1: Generate SSH Key Pair

On your **local machine** (not the server):

```bash
# Generate SSH key specifically for GitHub Actions
ssh-keygen -t ed25519 -C "github-actions@aiagents4biz.in" -f ~/.ssh/github_actions_deploy

# Press Enter to accept default location
# Press Enter twice (no passphrase, or set one if preferred)
```

### Step 2: Add Public Key to Server

```bash
# Copy public key to your server
ssh-copy-id -i ~/.ssh/github_actions_deploy.pub user@aiagents4biz.in

# Or manually:
cat ~/.ssh/github_actions_deploy.pub
# Then on server: echo "PASTE_KEY_HERE" >> ~/.ssh/authorized_keys
```

### Step 3: Get Private Key Content

```bash
# Display the private key (copy everything including BEGIN/END lines)
cat ~/.ssh/github_actions_deploy
```

**Important**: Copy the ENTIRE output, including:
```
-----BEGIN OPENSSH PRIVATE KEY-----
[all the key content]
-----END OPENSSH PRIVATE KEY-----
```

### Step 4: Add Secrets to GitHub

1. Go to your repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** for each secret:

#### Secret 1: `HOST_IP`
- **Name**: `HOST_IP`
- **Value**: `aiagents4biz.in`
- Click **Add secret**

#### Secret 2: `SSH_USER`
- **Name**: `SSH_USER`
- **Value**: Your SSH username (e.g., `ubuntu`, `root`)
- Click **Add secret**

#### Secret 3: `SSH_PRIVATE_KEY`
- **Name**: `SSH_PRIVATE_KEY`
- **Value**: Paste the entire private key from Step 3
- Click **Add secret**

#### Secret 4: `DEPLOY_PATH`
- **Name**: `DEPLOY_PATH`
- **Value**: `/var/www/oi-tracker`
- Click **Add secret**

### Step 5: Verify Setup

1. Check that all 4 secrets are listed in **Settings → Secrets and variables → Actions**
2. Push a commit to `main` branch
3. Go to **Actions** tab to see the workflow run
4. The workflow will validate secrets before attempting deployment

## Troubleshooting

### "HOST_IP secret is not set"
- Go to Settings → Secrets → Actions
- Verify `HOST_IP` exists and has value `aiagents4biz.in`

### "SSH_PRIVATE_KEY secret is not set"
- Verify the secret exists
- Make sure you copied the ENTIRE key including BEGIN/END lines
- Check for any extra spaces or line breaks

### "Permission denied (publickey)"
- Verify the public key was added to server's `~/.ssh/authorized_keys`
- Check file permissions on server:
  ```bash
  chmod 700 ~/.ssh
  chmod 600 ~/.ssh/authorized_keys
  ```

### "Failed to connect to server"
- Verify `HOST_IP` is correct (try: `ping aiagents4biz.in`)
- Verify `SSH_USER` is correct
- Test SSH connection manually:
  ```bash
  ssh -i ~/.ssh/github_actions_deploy user@aiagents4biz.in
  ```

## Security Notes

- ⚠️ **Never commit SSH keys to the repository**
- ⚠️ **Never share your private key**
- ✅ **Use separate SSH keys for GitHub Actions** (don't reuse your personal keys)
- ✅ **Rotate keys periodically**
- ✅ **Use ed25519 keys** (more secure than RSA)

