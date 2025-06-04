# 🚀 Enhanced Release Workflow System

## Overview
Enhanced release workflows for datason with target version support, GitHub tagging, and comprehensive validation.

## Features
- **🎯 Target Version Support**: Specify exact versions for marketing releases
- **🏷️ Automatic GitHub Tagging**: Creates and pushes Git tags
- **🎉 GitHub Release Creation**: Automated release creation
- **🔍 Comprehensive Validation**: Version format and collision detection
- **🔄 Force Release Option**: Override change detection
- **📊 Version Sync Checking**: Detects version misalignment

## Workflows
1. **🔨 Patch Bump**: Bug fixes (`x.y.z` → `x.y.z+1`)
2. **🔼 Minor Bump**: New features (`x.y.z` → `x.y+1.0`)
3. **🚀 Major Bump**: Breaking changes (`x.y.z` → `x+1.0.0`)

## Input Parameters
- `target_version` (optional): Custom version override
- `release_notes` (optional): Custom release notes
- `force_release` (optional): Force release without changes

## Target Version Validation
- **Patch**: Must increment patch only (`0.5.2` → `0.5.3`)
- **Minor**: Must increment minor only (`0.5.2` → `0.6.0`)
- **Major**: Must increment major only (`0.5.2` → `1.0.0`)

## Usage Examples

### Auto-increment
```
Actions → Bump Minor Version & Release
target_version: (leave empty)
Result: Auto-increments version
```

### Marketing Release
```
Actions → Bump Major Version & Release
target_version: "1.0.0"
release_notes: "🎉 Major milestone!"
Result: Creates v1.0.0
```

### Force Release
```
Actions → Bump Patch Version & Release
force_release: true
Result: Release without code changes
```

## File Updates
- `pyproject.toml`: Version field updated
- `datason/__init__.py`: `__version__` updated (if exists)

## Version Priority
1. PyPI version (if exists)
2. Git tag version
3. Project file version

## Safety Features
- Prevents duplicate tag creation
- Validates version increment logic
- Checks for code changes
- Warns about version misalignment
- Creates GitHub releases automatically

## Troubleshooting
- **Tag exists**: Use different target version
- **No changes**: Use `force_release: true`
- **Invalid version**: Check increment logic
