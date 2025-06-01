# Release Management Guide for datason

## 🏷️ Overview

datason uses a comprehensive release management system with automated workflows, semantic versioning, and strict quality controls. This guide covers the entire process from development to release.

## 📋 Table of Contents

- [Branch Strategy](#branch-strategy)
- [Version Management](#version-management)
- [Pull Request Workflow](#pull-request-workflow)
- [Release Process](#release-process)
- [Automation Features](#automation-features)
- [Troubleshooting](#troubleshooting)

## 🌲 Branch Strategy

### **Main Branch (`main`)**
- **Purpose**: Production-ready code
- **Protection**: Fully protected, requires PRs
- **CI/CD**: Full pipeline runs on every push
- **Deploy**: Triggers documentation updates

### **Feature Branches**
- **Naming**: `feature/description`, `fix/issue-number`, `docs/update-name`
- **Lifecycle**: Created → PR → Review → Merge → Delete
- **Requirements**: Must pass all CI checks

### **Branch Protection Rules**
```yaml
Required status checks:
  ✅ 🔍 Code Quality & Security  
  ✅ 🧪 CI / test (ubuntu-latest, 3.11)
  ✅ 📚 Build Documentation

Restrictions:
  ✅ Require PR reviews (1 minimum)
  ✅ Dismiss stale reviews  
  ✅ Require conversation resolution
  ✅ Include administrators
  ✅ Linear history preferred
```

## 📦 Version Management

### **Semantic Versioning**
datason follows [Semantic Versioning 2.0.0](https://semver.org/):

```
v{MAJOR}.{MINOR}.{PATCH}[-{PRERELEASE}]

Examples:
✅ v0.1.0     - Initial release
✅ v0.1.1     - Patch release (bug fixes)
✅ v0.2.0     - Minor release (new features)
✅ v1.0.0     - Major release (breaking changes)
✅ v1.0.0-rc.1 - Release candidate
✅ v1.0.0-beta.2 - Beta release
```

### **Version Increment Guidelines**

| Change Type | Version | Example |
|-------------|---------|---------|
| 🐛 **Bug fixes** | PATCH | v0.1.0 → v0.1.1 |
| ✨ **New features** | MINOR | v0.1.1 → v0.2.0 |
| 💥 **Breaking changes** | MAJOR | v0.2.0 → v1.0.0 |
| 🧪 **Pre-releases** | PRERELEASE | v1.0.0-beta.1 |

### **Release Schedule**
- **Patch releases**: Weekly (bug fixes, docs)
- **Minor releases**: Monthly (new features)
- **Major releases**: Quarterly (breaking changes)
- **Security releases**: As needed (critical fixes)

## 🔄 Pull Request Workflow

### **Creating a PR**

1. **Create feature branch**:
   ```bash
   git checkout -b feature/awesome-feature
   git push -u origin feature/awesome-feature
   ```

2. **Follow PR template**: Our template guides you through all requirements

3. **Ensure quality**:
   ```bash
   # Run locally before creating PR
   pre-commit run --all-files
   pytest --cov=datason
   mkdocs build --strict
   ```

### **PR Requirements**

#### **Automatic Checks**
- ✅ **Quality Pipeline**: Ruff linting, formatting, security scanning
- ✅ **CI Pipeline**: Tests across Python 3.8-3.13, coverage analysis
- ✅ **Documentation**: MkDocs builds successfully
- ✅ **Pre-commit**: All hooks pass

#### **Manual Review Checklist**
- ✅ Code follows project guidelines
- ✅ Tests added for new functionality
- ✅ Documentation updated
- ✅ CHANGELOG.md updated
- ✅ Backward compatibility maintained
- ✅ Performance impact assessed

### **Auto-merge Eligibility**

**✅ Automatically merged when**:
- All CI checks pass
- PR has `auto-merge` label
- Minor/patch dependency updates
- Documentation-only changes
- Non-breaking CI improvements

**❌ Manual review required for**:
- Breaking changes
- Major version updates
- Core functionality changes
- Security-related changes

### **PR Labels System**

#### **Type Labels**
- `bug` - Bug fixes
- `enhancement` - New features
- `documentation` - Documentation updates
- `security` - Security improvements
- `performance` - Performance optimizations

#### **Priority Labels**
- `priority:critical` - Immediate attention required
- `priority:high` - High priority
- `priority:medium` - Medium priority
- `priority:low` - Low priority

#### **Status Labels**
- `status:needs-review` - Awaiting code review
- `status:ready-to-merge` - Approved and ready
- `auto-merge` - Auto-merge when checks pass

#### **Component Labels**
- `component:core` - Core serialization
- `component:ml` - ML integrations
- `component:docs` - Documentation
- `component:ci` - CI/CD changes

## 🚀 Release Process

### **Automated Release Workflow**

#### **Method 1: Tag-based Release**
```bash
# Create and push tag
git tag v0.1.2
git push origin v0.1.2

# This triggers:
# 1. ✅ Version validation
# 2. 📝 Changelog generation  
# 3. 🏷️ GitHub release creation
# 4. 📦 PyPI publication
```

#### **Method 2: Manual Release**
```bash
# Use GitHub CLI
gh workflow run release.yml \
  --field version=v0.1.2 \
  --field prerelease=false
```

#### **Method 3: GitHub UI**
1. Go to **Actions** → **🏷️ Release Management**
2. Click **Run workflow**
3. Enter version (e.g., `v0.1.2`)
4. Select if pre-release
5. Click **Run workflow**

### **Release Checklist**

#### **Pre-release**
- [ ] All tests pass on main branch
- [ ] Documentation is up to date
- [ ] CHANGELOG.md reflects all changes
- [ ] Version number follows semantic versioning
- [ ] Breaking changes are documented

#### **During Release**
- [ ] Automated release workflow completes
- [ ] GitHub release is created with notes
- [ ] PyPI package is published successfully
- [ ] Documentation is updated

#### **Post-release**
- [ ] Release announcement (if major/minor)
- [ ] Social media updates (if significant)
- [ ] Community discussions created
- [ ] Monitor for issues

### **Release Notes Format**

Our automated system generates release notes like:

```markdown
## 🎉 What's New in v0.1.2

- ✨ Added support for TensorFlow 2.15
- 🐛 Fixed circular reference detection
- 📚 Updated API documentation
- ⚡ Improved serialization performance by 15%

## 📦 Installation

pip install datason==v0.1.2

## 🔗 Links
- 📚 Documentation: https://datason.readthedocs.io
- 🐛 Issues: https://github.com/danielendler/datason/issues
- 💬 Discussions: https://github.com/danielendler/datason/discussions
```

## 🤖 Automation Features

### **Dependabot Integration**
- **Weekly dev dependency updates** (pytest, ruff, mypy)
- **Monthly core dependency updates** (pandas, numpy)
- **Conservative ML library updates** (manual review required)
- **Automatic labeling and categorization**

### **Auto-merge Criteria**
```yaml
Automatically merged:
  ✅ Dependabot minor/patch updates
  ✅ Documentation-only changes
  ✅ CI configuration improvements
  ✅ Test additions/improvements
  ✅ Code formatting changes

Manual review required:
  ❌ Major version updates
  ❌ Breaking changes
  ❌ Core functionality changes
  ❌ Security-related changes
```

### **Quality Gates**
Every PR must pass:
1. **🔍 Code Quality**: Ruff linting, formatting, security scanning
2. **🧪 Testing**: All tests across Python 3.8-3.13
3. **📊 Coverage**: Maintain 80%+ code coverage
4. **📚 Documentation**: MkDocs builds without errors
5. **🔒 Security**: No high/medium severity issues

## 🛠️ Setup Instructions

### **For Repository Administrators**

1. **Set up branch protection**:
   ```bash
   # Install GitHub CLI if needed
   gh auth login

   # Run our setup script
   python scripts/setup_github_labels.py
   ```

2. **Configure repository settings**:
   - Enable Discussions
   - Set up security alerts
   - Configure merge options (squash preferred)

3. **Set up secrets** (for PyPI publishing):
   ```
   PYPI_API_TOKEN - PyPI API token for publishing
   ```

### **For Contributors**

1. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

2. **Configure Git**:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

3. **Follow the workflow**:
   - Create feature branch
   - Make changes
   - Run tests locally
   - Create PR using template
   - Address review feedback

## 🔧 Troubleshooting

### **Common Issues**

#### **PR Auto-merge Not Working**
```bash
# Check if PR has auto-merge label
gh pr view --json labels

# Check CI status
gh pr checks

# Manually trigger auto-merge
gh pr merge --auto --squash
```

#### **Release Failed to Publish**
```bash
# Check workflow status
gh run list --workflow=release.yml

# View specific run logs
gh run view <run-id>

# Manual PyPI publish (if needed)
python -m build
twine upload dist/*
```

#### **Version Tag Issues**
```bash
# List existing tags
git tag -l

# Delete incorrect tag (locally and remote)
git tag -d v0.1.2
git push origin :refs/tags/v0.1.2

# Create correct tag
git tag v0.1.2
git push origin v0.1.2
```

### **Emergency Procedures**

#### **Hotfix Release**
```bash
# Create hotfix branch from main
git checkout main
git pull origin main
git checkout -b hotfix/critical-fix

# Make minimal fix
# Test thoroughly
# Create PR with priority:critical label
# Emergency merge after review
# Tag and release immediately
```

#### **Rollback Release**
```bash
# Mark GitHub release as pre-release
gh release edit v0.1.2 --prerelease

# Yank from PyPI if needed (contact maintainers)
# Create fixed version immediately
```

## 📚 Additional Resources

- [Contributing Guide](CONTRIBUTING.md)
- [CI/CD Pipeline Guide](CI_PIPELINE_GUIDE.md)
- [CI Performance Optimization](CI_PERFORMANCE.md)
- [Pipeline Audit Report](PIPELINE_AUDIT.md)
- [Plugin Testing Strategy](PLUGIN_TESTING.md)
- [Tooling Guide](TOOLING_GUIDE.md)
- [Dependabot Setup](DEPENDABOT_GUIDE.md)
- [GitHub Pages Setup](GITHUB_PAGES_SETUP.md)
- [Security Policy](SECURITY.md)
- [GitHub Flow Documentation](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Semantic Versioning Specification](https://semver.org/)

---

> 💡 **Questions?** Create a [discussion](https://github.com/danielendler/datason/discussions) or check existing [issues](https://github.com/danielendler/datason/issues).

## Additional Documentation

- [Contributing Guide](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [CI/CD Pipeline](CI_PIPELINE_GUIDE.md)
