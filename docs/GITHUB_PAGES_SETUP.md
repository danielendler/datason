# GitHub Pages Setup Guide for datason

## 🚀 Overview

This guide explains how to set up GitHub Pages for datason documentation and resolve common permission issues with GitHub Actions deployment.

## 🔧 Repository Setup

### **Step 1: Enable GitHub Pages**

1. Go to your repository settings: `https://github.com/danielendler/datason/settings`
2. Scroll down to **"Pages"** section
3. Under **"Source"**, select **"GitHub Actions"** (not the legacy "Deploy from a branch")
4. Click **"Save"**

### **Step 2: Configure Repository Permissions**

1. Go to **Settings** → **Actions** → **General**
2. Under **"Workflow permissions"**, select:
   - ✅ **"Read and write permissions"**
   - ✅ **"Allow GitHub Actions to create and approve pull requests"**
3. Click **"Save"**

### **Step 3: Verify Environment Protection** (Optional)

1. Go to **Settings** → **Environments**
2. If `github-pages` environment exists, ensure it's configured correctly
3. If not, it will be created automatically on first deployment

## 🏗️ Modern GitHub Pages Workflow

Our updated `.github/workflows/docs.yml` now uses the modern GitHub Pages workflow with:

### **Key Improvements**
- ✅ **Official GitHub Actions**: Uses `actions/configure-pages@v4` and `actions/deploy-pages@v4`
- ✅ **Proper Permissions**: Workflow-level permissions for Pages deployment
- ✅ **Concurrency Control**: Prevents multiple simultaneous deployments
- ✅ **Environment Protection**: Uses `github-pages` environment for deployment tracking

### **Workflow Features**
```yaml
# Workflow-level permissions (most secure)
permissions:
  contents: read
  pages: write
  id-token: write

# Concurrency control
concurrency:
  group: "pages"
  cancel-in-progress: false

# Environment for deployment tracking
environment:
  name: github-pages
  url: ${{ steps.deployment.outputs.page_url }}
```

## 🐛 Troubleshooting Common Issues

### **1. Permission Denied Error**
```
remote: Permission to danielendler/datason.git denied to github-actions[bot].
fatal: unable to access 'https://github.com/danielendler/datason.git/': The requested URL returned error: 403
```

**Solution:**
- ✅ Enable "Read and write permissions" in repository settings
- ✅ Use the modern GitHub Pages workflow (not `peaceiris/actions-gh-pages`)
- ✅ Set GitHub Pages source to "GitHub Actions"

### **2. Environment Protection Rules**
```
Error: The environment 'github-pages' is protected and cannot be used by this workflow.
```

**Solution:**
- ✅ Check **Settings** → **Environments** → **github-pages**
- ✅ Ensure workflow has proper permissions
- ✅ Add required reviewers if needed for protection

### **3. Pages Build Failure**
```
Error: Failed to create deployment
```

**Solution:**
- ✅ Ensure Pages source is set to "GitHub Actions"
- ✅ Check that `mkdocs build --strict` passes locally
- ✅ Verify all documentation links are valid

### **4. Workflow Not Triggering**
```
Workflow doesn't run on documentation changes
```

**Solution:**
- ✅ Check path filters in workflow trigger:
  ```yaml
  paths: ['docs/**', 'mkdocs.yml', 'README.md']
  ```
- ✅ Ensure changes are in tracked paths
- ✅ Push to `main` branch for deployment

## 📋 Workflow Breakdown

### **Build Stage**
```yaml
build-docs:
  # Runs on all PRs and pushes
  # Validates documentation builds correctly
  # Uploads artifact for later deployment
```

### **Deploy Stage**
```yaml
deploy-github-pages:
  # Only runs on main branch pushes
  # Uses modern GitHub Pages actions
  # Deploys to github-pages environment
```

## 🚀 Deployment Process

### **Automatic Deployment**
1. **Push to main** with documentation changes
2. **Build job** validates and creates artifact
3. **Deploy job** publishes to GitHub Pages
4. **Site available** at `https://danielendler.github.io/datason/`

### **Manual Deployment**
```bash
# Trigger workflow manually
gh workflow run docs.yml
```

## 🔍 Monitoring & Debugging

### **Check Deployment Status**
```bash
# View workflow runs
gh run list --workflow=docs.yml

# View specific run details
gh run view <run-id>

# Watch live run
gh run watch
```

### **Verify Pages Configuration**
```bash
# Check Pages settings via API
gh api repos/danielendler/datason/pages
```

### **Test Documentation Locally**
```bash
# Install dependencies
pip install -e ".[docs]"

# Serve locally
mkdocs serve

# Build and check
mkdocs build --strict
```

## 📚 Alternative Solutions

### **Option 1: Personal Access Token** (if above doesn't work)
1. Create PAT with `repo` and `workflow` permissions
2. Add as repository secret `PAGES_DEPLOY_TOKEN`
3. Use in workflow:
   ```yaml
   - uses: peaceiris/actions-gh-pages@v3
     with:
       personal_token: ${{ secrets.PAGES_DEPLOY_TOKEN }}
       publish_dir: ./site
   ```

### **Option 2: GitHub App Token**
For organization repositories with stricter security:
1. Create GitHub App with Pages permissions
2. Install app on repository
3. Use app token for deployment

## 🔐 Security Considerations

### **Minimal Permissions**
Our workflow uses minimal required permissions:
- `contents: read` - Read repository content
- `pages: write` - Deploy to GitHub Pages
- `id-token: write` - OIDC token for Pages deployment

### **Environment Protection**
- Deployments tracked in `github-pages` environment
- Optional: Add required reviewers for production
- Automatic deployment logs and history

## 📖 Additional Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions for Pages](https://github.com/actions/deploy-pages)
- [MkDocs Documentation](https://www.mkdocs.org/)
- [datason Documentation Source](https://github.com/danielendler/datason/tree/main/docs)

---

> 💡 **Need Help?** Create an [issue](https://github.com/danielendler/datason/issues) if you encounter problems with documentation deployment.
