# Security Policy

## Security Philosophy

datason prioritizes security alongside performance when handling Python object serialization. This document outlines our security practices, potential risks, and recommended usage patterns.

## Security Status

✅ **Low Risk** - datason has been hardened with real security protections against common JSON serialization vulnerabilities.

**Last Security Audit**: 2025-05-30  
**Security Scanner Results**: ✅ 1 minor issue (documented), ✅ 0 critical vulnerabilities  
**Dependencies**: ✅ All patched to latest secure versions

## Supported Versions

| Version | Supported          | Security Features |
| ------- | ------------------ | ----------------- |
| 0.1.x   | ✅ **Current**     | Full protection   |

## 🛡️ Built-in Security Protections

### **1. Circular Reference Detection**
**Real Protection**: Prevents infinite recursion and memory exhaustion.

```python
import datason

# This data structure would crash other serializers
a = {}
b = {"a": a}
a["b"] = b  # Circular reference

# datason handles it safely
result = datason.serialize(a)
# Warns: "Circular reference detected. Replacing with null to prevent infinite recursion."
# Returns: {"b": {"a": None}}  # Safe, controlled output
```

**How it works**: Tracks object IDs during serialization to detect cycles.

### **2. Resource Exhaustion Prevention**
**Real Protection**: Enforces limits to prevent DoS attacks.

```python
# These would be rejected with SecurityError:
huge_dict = {f"key_{i}": i for i in range(20_000_000)}  # > 10M items
deep_nesting = create_nested_dict(depth=2000)  # > 1K depth
massive_string = "x" * 2_000_000  # > 1M chars (gets truncated)

try:
    datason.serialize(huge_dict)
except datason.SecurityError as e:
    print(f"Blocked: {e}")
    # Output: "Dictionary size (20000000) exceeds maximum (10000000).
    #          This may indicate a resource exhaustion attempt."
```

**Security Limits**:
- **Max Object Size**: 10,000,000 items (dictionaries, lists, arrays)
- **Max Recursion Depth**: 1,000 levels (prevents stack overflow)
- **Max String Length**: 1,000,000 characters (truncated with warning)

### **3. Safe Error Handling**
**Real Protection**: No information leakage through error messages.

```python
class ProblematicObject:
    def __dict__(self):
        raise RuntimeError("Internal error with sensitive data")

# datason handles safely
obj = ProblematicObject()
result = datason.serialize(obj)
# Warns: "Failed to serialize object. Falling back to string representation."
# Returns: Safe fallback, no sensitive data exposed
```

### **4. Input Validation**
**Real Protection**: Type checking and safe handling of all input types.

- ✅ **No arbitrary code execution** (unlike `pickle`)
- ✅ **Controlled type handling** for all supported data types
- ✅ **Safe fallbacks** for unknown objects
- ✅ **Memory-safe operations** for large datasets

## 🔍 Security Validation Results

### **Bandit Security Scan** - ✅ PASSED
```
loc: 1,082 lines of code scanned
SEVERITY.HIGH: 0
SEVERITY.MEDIUM: 0  
SEVERITY.LOW: 1 (intentional, documented)
```

**Only Issue**: One intentional `try-except-pass` for handling edge cases where `hasattr()` fails. This is documented and safe.

### **Dependency Vulnerabilities** - ✅ RESOLVED
**Recent Actions**:
- ✅ Updated `jinja2` from 3.1.4 → 3.1.6 (fixed 3 CVEs)
- ✅ Updated `setuptools` from 70.2.0 → 80.9.0 (fixed path traversal CVE)

**Dependency Strategy**:
- Core datason has **zero dependencies** for security
- Optional dependencies (pandas, numpy, ML libraries) are user-controlled
- All dev dependencies regularly updated and scanned

### **Real-World Attack Prevention**

| Attack Vector | Protection | Status |
|---------------|------------|--------|
| **Billion Laughs (XML bomb equivalent)** | Size limits + depth limits | ✅ Protected |
| **Memory exhaustion** | Resource limits on all data types | ✅ Protected |
| **Stack overflow** | Recursion depth tracking | ✅ Protected |
| **Information leakage** | Safe error handling + logging | ✅ Protected |
| **Circular reference DoS** | Object ID tracking | ✅ Protected |

## 🚨 Reporting Security Issues

**Please DO NOT report security vulnerabilities through public GitHub issues.**

### Preferred: Security Advisory
1. Go to https://github.com/danielendler/datason/security/advisories
2. Click "Report a vulnerability"
3. Provide details including reproduction steps

### Alternative: Email
📧 **security@datason.dev**

**Include in your report:**
- Description and impact assessment
- Minimal reproduction example
- Your environment details
- Suggested fix (if you have one)

### Response Timeline

| Timeframe | Our Commitment |
|-----------|----------------|
| **24 hours** | Acknowledgment |
| **72 hours** | Initial assessment |
| **1 week** | Investigation complete |
| **2 weeks** | Fix deployed (if valid) |

## 🔒 Production Security Best Practices

### **Environment Setup**
```bash
# Install with security scanning
pip install datason[dev]
bandit -r your_project/
safety scan
```

### **Secure Usage Patterns**
```python
import datason

# ✅ GOOD: Handle untrusted data safely
try:
    result = datason.serialize(untrusted_data)
except datason.SecurityError as e:
    logger.warning(f"Blocked potentially malicious data: {e}")
    return None

# ✅ GOOD: Monitor for warnings in production
import warnings
with warnings.catch_warnings(record=True) as w:
    result = datason.serialize(data)
    if w:
        logger.info(f"Security warnings: {[str(warning.message) for warning in w]}")

# ❌ AVOID: Don't serialize sensitive data
sensitive_data = {"password": "secret", "api_key": "12345"}
# Filter before serializing
safe_data = {k: v for k, v in data.items() if k not in ["password", "api_key"]}
result = datason.serialize(safe_data)
```

### **Monitoring & Alerting**
```python
# Set up security monitoring
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("datason.security")

# This will capture security warnings in your logs
warnings.filterwarnings("always", category=UserWarning, module="datason")
```

## 📋 Security Configuration

### **Runtime Security Settings**
```python
# Security constants (configurable in future versions)
from datason.core import MAX_SERIALIZATION_DEPTH, MAX_OBJECT_SIZE, MAX_STRING_LENGTH

print(f"Max depth: {MAX_SERIALIZATION_DEPTH}")      # 1,000
print(f"Max object size: {MAX_OBJECT_SIZE}")        # 10,000,000
print(f"Max string length: {MAX_STRING_LENGTH}")    # 1,000,000
```

### **Recommended CI/CD Security Checks**
```yaml
# .github/workflows/security.yml
- name: Security Scan
  run: |
    pip install bandit safety
    bandit -r datason/
    safety scan

- name: Dependency Audit  
  run: pip-audit
```

## 🏆 Security Achievements

- ✅ **Zero critical vulnerabilities** in current release
- ✅ **Proactive security design** with built-in protections
- ✅ **Comprehensive test coverage** for security features
- ✅ **Regular security updates** of dependencies
- ✅ **Transparent security practices** and open auditing

## 📚 Security Resources

- [OWASP JSON Security](https://owasp.org/www-project-json-sanitizer/)
- [Python Security Best Practices](https://python.org/dev/security/)
- [Secure Coding Guidelines](https://wiki.sei.cmu.edu/confluence/display/python)
- [CVE Database](https://cve.mitre.org/)

---

**🛡️ Security is a continuous process.** Help us keep datason secure by reporting issues responsibly and following security best practices in your own code.
# Test
