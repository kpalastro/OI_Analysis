# How to Clone Repository on Server

If you're getting "Permission denied (publickey)" error, you have two options:

## Option 1: Use HTTPS (Easier - Recommended for Initial Setup)

```bash
# Clone using HTTPS (will prompt for GitHub username/password or token)
git clone https://github.com/kpalastro/OI_Analysis.git /var/www/oi-tracker

# Or if you need a personal access token:
# Go to GitHub → Settings → Developer settings → Personal access tokens → Generate new token
# Then use: git clone https://YOUR_TOKEN@github.com/kpalastro/OI_Analysis.git /var/www/oi-tracker
```

## Option 2: Set Up SSH Keys (Better for CI/CD)

### Step 1: Generate SSH Key on Server

```bash
# Generate SSH key pair
ssh-keygen -t ed25519 -C "server@aiagents4biz.in"
# Press Enter to accept default location (~/.ssh/id_ed25519)
# Press Enter twice for no passphrase (or set one if preferred)

# Display public key
cat ~/.ssh/id_ed25519.pub
```

### Step 2: Add SSH Key to GitHub

1. Copy the public key output from above
2. Go to GitHub → Settings → SSH and GPG keys
3. Click "New SSH key"
4. Paste your public key
5. Click "Add SSH key"

### Step 3: Test SSH Connection

```bash
ssh -T git@github.com
# Should say: "Hi kpalastro! You've successfully authenticated..."
```

### Step 4: Clone Repository

```bash
git clone git@github.com:kpalastro/OI_Analysis.git /var/www/oi-tracker
```

## Option 3: Use Personal Access Token (HTTPS)

If you prefer HTTPS but want to avoid password prompts:

### Step 1: Generate Personal Access Token

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name: "OI Tracker Server"
4. Select scopes: `repo` (full control of private repositories)
5. Click "Generate token"
6. **Copy the token immediately** (you won't see it again!)

### Step 2: Clone with Token

```bash
# Replace YOUR_TOKEN with the token you copied
git clone https://YOUR_TOKEN@github.com/kpalastro/OI_Analysis.git /var/www/oi-tracker

# Or set it as environment variable
export GITHUB_TOKEN=your_token_here
git clone https://${GITHUB_TOKEN}@github.com/kpalastro/OI_Analysis.git /var/www/oi-tracker
```

### Step 3: Configure Git Credential Helper (Optional)

To avoid entering token every time:

```bash
# Store credentials
git config --global credential.helper store

# Or use cache (credentials stored for 15 minutes)
git config --global credential.helper cache
```

## Quick Fix for Current Error

If you just want to get started quickly:

```bash
# Use HTTPS instead
cd /var/www
sudo rm -rf oi-tracker  # Remove if exists
git clone https://github.com/kpalastro/OI_Analysis.git oi-tracker
cd oi-tracker
```

## For CI/CD (GitHub Actions)

The CI/CD pipeline uses SSH keys stored in GitHub Secrets. You'll need to:

1. Generate a separate SSH key pair for GitHub Actions
2. Add the **public key** to your server's `~/.ssh/authorized_keys`
3. Add the **private key** to GitHub Secrets as `SSH_PRIVATE_KEY`

See `.github/workflows/README.md` for details.

