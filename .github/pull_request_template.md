# Pull Request for datason

## 🎯 **What does this PR do?**
<!-- Provide a clear, concise description of the changes -->

## 📋 **Type of Change**
<!-- Check all that apply -->
- [ ] 🐛 **Bug fix** (non-breaking change that fixes an issue)
- [ ] ✨ **New feature** (non-breaking change that adds functionality)
- [ ] 💥 **Breaking change** (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📚 **Documentation** (updates to docs, README, etc.)
- [ ] 🧪 **Tests** (adding missing tests or correcting existing tests)
- [ ] 🔧 **CI/DevOps** (changes to build process, CI configuration, etc.)
- [ ] 🎨 **Code style** (formatting, renaming, etc. - no functional changes)
- [ ] ♻️ **Refactoring** (code changes that neither fix bugs nor add features)
- [ ] ⚡ **Performance** (changes that improve performance)
- [ ] 🔒 **Security** (security-related changes)

## 🔗 **Related Issues**
<!-- Link related issues: Fixes #123, Closes #456, Related to #789 -->

## ✅ **Checklist**
<!-- Mark completed items with [x] -->

### Code Quality
- [ ] Code follows project style guidelines (ruff passes)
- [ ] Self-review of code completed
- [ ] Code is well-commented and documented
- [ ] No debug statements or console.log left in code

### Testing
- [ ] New tests added for new functionality
- [ ] Existing tests updated if necessary
- [ ] All tests pass locally (`pytest`)
- [ ] Coverage maintained or improved (`pytest --cov=datason`)

### Documentation
- [ ] Documentation updated (if user-facing changes)
- [ ] README.md updated (if necessary)
- [ ] CHANGELOG.md updated with changes
- [ ] API documentation updated (if applicable)

### Compatibility
- [ ] Changes are backward compatible
- [ ] Breaking changes documented and justified
- [ ] Dependency changes are minimal and justified

### Optional Dependencies
<!-- Check if your changes affect optional dependencies -->
- [ ] pandas integration tested (if applicable)
- [ ] numpy integration tested (if applicable)  
- [ ] ML libraries tested (if applicable)
- [ ] Works without optional dependencies

## 🧪 **Testing**
<!-- Describe the testing you performed -->

### Test Environment
- **Python version(s)**:
- **Operating System**:
- **Dependencies**:

### Test Coverage
<!-- Paste relevant test output or describe test scenarios -->
```bash
# Example:
$ pytest tests/test_your_feature.py -v
$ pytest --cov=datason --cov-report=term-missing
```

## 📊 **Performance Impact**
<!-- If performance-related, include benchmarks -->
<!-- Use: just perf-real / just perf-real-compare -->
- **Hotspot(s) touched**:
- **Workload classes impacted** (`api`, `event_log`, `data_science`, `ml_serving`):
- **Real-data replay p95 delta**:
- **Real-data replay p99 delta**:
- **Throughput delta**:
- **Peak memory delta**:
- **Synthetic benchmark impact (secondary signal)**:

## 📸 **Screenshots/Examples**
<!-- For UI changes or new features, include examples -->

## 🔄 **Migration Guide**
<!-- For breaking changes, provide migration instructions -->

## 📝 **Additional Notes**
<!-- Any additional information, concerns, or discussion points -->

---

## 🤖 **For Maintainers**

### Auto-merge Eligibility
<!-- Maintainers: check if this PR qualifies for auto-merge -->
- [ ] **Dependencies**: Minor/patch dependency updates
- [ ] **Documentation**: Documentation-only changes  
- [ ] **CI/Build**: Non-breaking CI improvements
- [ ] **Tests**: Test additions/improvements
- [ ] **Formatting**: Code style/formatting only

### Review Priority
- [ ] **High**: Breaking changes, security fixes, major features
- [ ] **Medium**: New features, significant improvements
- [ ] **Low**: Documentation, tests, minor fixes

---

**📚 Need help?** Check our [Contributing Guide](https://github.com/danielendler/datason/blob/main/CONTRIBUTING.md) | [Development Setup](https://github.com/danielendler/datason/blob/main/CONTRIBUTING.md#development-setup)
