# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2025-05-30

### 🚀 Major Changes
- **BREAKING**: Renamed package from `serialpy` to `datason`
  - Updated all imports: `from serialpy` → `from datason`
  - Updated package name in PyPI and documentation
  - Maintained full API compatibility

### 🔧 DevOps Infrastructure Fixes
- **Fixed GitHub Pages deployment permission errors**
  - Replaced deprecated `peaceiris/actions-gh-pages@v3` with modern GitHub Pages workflow
  - Updated to use official actions: `actions/configure-pages@v4`, `actions/upload-pages-artifact@v3`, `actions/deploy-pages@v4`
  - Added proper workflow-level permissions: `contents: read`, `pages: write`, `id-token: write`
  - Added concurrency control and `workflow_dispatch` trigger

- **Resolved Dependabot configuration issues**
  - Fixed missing labels error by adding 43+ comprehensive labels
  - Fixed overlapping directories error by consolidating pip configurations
  - Replaced deprecated reviewers field with `.github/CODEOWNERS` file
  - Changed target branch from `develop` to `main`

- **Fixed auto-merge workflow circular dependency**
  - Resolved infinite loop where auto-merge waited for itself
  - Updated to `pull_request_target` event with proper permissions
  - Upgraded to `hmarr/auto-approve-action@v4`
  - Added explicit `pull-requests: write` permissions
  - Fixed "fatal: not a git repository" errors with proper checkout steps
  - Updated to `fastify/github-action-merge-dependabot@v3` with correct configuration

### 📊 Improved Test Coverage & CI
- **Enhanced Codecov integration**
  - Fixed coverage reporting from 45% to 86% by running all test files
  - Added Codecov test results action with JUnit XML
  - Added proper CODECOV_TOKEN usage and branch coverage tracking
  - Upload HTML coverage reports as artifacts

- **Fixed PyPI publishing workflow**
  - Updated package name references from 'pyjsonify' to 'datason'
  - Fixed trusted publishing configuration
  - Added comprehensive build verification and package checks

### 📚 Documentation Updates
- **Updated contributing guidelines**
  - Replaced black/flake8 references with ruff
  - Updated development workflow from 8 to 7 steps: `ruff check --fix . && ruff format .`
  - Updated code style description to "Linter & Formatter: Ruff (unified tool)"

- **Enhanced repository configuration**
  - Added comprehensive repository description and topics
  - Updated website links to reflect new package name
  - Added comprehensive setup guide at `docs/GITHUB_PAGES_SETUP.md`

### 🔒 Security & Branch Protection
- **Implemented smart branch protection**
  - Added GitHub repository rules requiring status checks and human approval
  - Enabled auto-merge for repository with proper protection rules
  - Configured branch protection to allow Dependabot bypass while maintaining security

### 🏷️ Release Management
- **Professional release workflow**
  - Updated version management from "0.1.0" to "0.1.3"
  - Implemented proper Git tagging and release automation
  - Added comprehensive changelog documentation
  - Fixed release date accuracy (2025-05-30)

### 🤖 Automation Features
- **Comprehensive workflow management**
  - Smart Dependabot updates: conservative for ML libraries, aggressive for dev tools
  - Complete labeling system with 43+ labels for categorization
  - Automated reviewer assignment via CODEOWNERS
  - Working auto-approval and auto-merge for safe dependency updates

### 🐛 Bug Fixes
- Fixed auto-merge semver warning with proper `target: patch` parameter
- Resolved GitHub Actions permission errors across all workflows
- Fixed environment validation errors in publishing workflows
- Corrected package import statements and references throughout codebase

### ⚡ Performance Improvements
- Optimized CI workflow to run all tests for complete coverage
- Enhanced build process with proper artifact management
- Improved development setup with updated tooling configuration

### 👨‍💻 Developer Experience
- Enhanced debugging with comprehensive logging and error reporting
- Improved development workflow with unified ruff configuration
- Better IDE support with updated type checking and linting
- Streamlined release process with automated tagging and publishing

## [0.1.1] - 2025-05-29

### Added
- Initial package structure and core serialization functionality
- Basic CI/CD pipeline setup
- Initial documentation framework
