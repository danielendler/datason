# Module-Based Test Consolidation

## Problem: Too Many Files Per Module

Each datason module has tests scattered across multiple files. We need **max 2-3 files per module**.

## Current Situation Analysis

### `datason.core` module (WORST: 10+ files!)
```
tests/unit/test_core_comprehensive.py              ✅ KEEP (main functionality)
tests/coverage/test_core_edge_cases.py             ✅ KEEP (edge cases)
tests/core/test_security.py                        🔄 MERGE → test_core_edge_cases.py
tests/core/test_ultra_fast_path_coverage.py        🔄 MERGE → test_core_edge_cases.py
tests/core/test_edge_cases.py                      🔄 MERGE → test_core_edge_cases.py
tests/core/test_circular_references.py             🔄 MERGE → test_core_edge_cases.py
tests/security/test_security_attack_vectors.py     🔄 MERGE → test_core_edge_cases.py
tests/features/test_chunked_streaming.py           🔄 MOVE → integration/
tests/coverage/test_targeted_coverage_boost.py     🔄 MERGE → test_core_edge_cases.py
+ others...
```

### `datason.deserializers` module (8+ files!)
```
tests/unit/test_deserializers_comprehensive.py     ✅ KEEP (main functionality)
tests/integration/test_template_deserializer.py    🔄 KEEP in integration (cross-module)
tests/features/test_template_deserialization.py    🔄 MERGE → deserializers_comprehensive.py
tests/enhanced_types/test_*_auto_detection.py      🔄 CREATE → test_deserializers_edge_cases.py
tests/coverage/test_lazy_imports_and_hotpath.py    🔄 CREATE → test_deserializers_edge_cases.py
+ others...
```

### `datason.api` module (looks good!)
```
tests/unit/test_api_comprehensive.py               ✅ KEEP (main functionality)
```

### `datason.config` module (looks good!)
```
tests/unit/test_config_comprehensive.py            ✅ KEEP (main functionality)
```

## Target Organization (2-3 files max per module)

### Core Module (datason.core):
```
tests/unit/test_core_comprehensive.py              # Main functionality
tests/coverage/test_core_edge_cases.py             # All edge cases merged here
```

### Deserializers Module (datason.deserializers):
```
tests/unit/test_deserializers_comprehensive.py     # Main functionality  
tests/coverage/test_deserializers_edge_cases.py    # Edge cases & auto-detection
```

### API Module (datason.api):
```
tests/unit/test_api_comprehensive.py               # Main functionality (already good)
```

### Config Module (datason.config):
```
tests/unit/test_config_comprehensive.py            # Main functionality (already good)
```

### Other Modules:
```
tests/unit/test_serializers_comprehensive.py       # datason.serializers
tests/unit/test_validation_comprehensive.py        # datason.validation  
tests/unit/test_converters_comprehensive.py        # datason.converters
tests/unit/test_data_utils_comprehensive.py        # datason.data_utils
```

## Action Plan

### Phase 1: Consolidate Core Module Tests
**Target**: Merge 6+ core edge case files into `test_core_edge_cases.py`

**Files to merge INTO `tests/coverage/test_core_edge_cases.py`:**
- `tests/core/test_security.py` (security edge cases)
- `tests/core/test_ultra_fast_path_coverage.py` (performance paths)
- `tests/core/test_edge_cases.py` (object edge cases)
- `tests/core/test_circular_references.py` (circular reference handling)
- `tests/security/test_security_attack_vectors.py` (security attacks)
- Functions from `tests/coverage/test_targeted_coverage_boost.py`

### Phase 2: Consolidate Deserializers Module Tests
**Target**: Create `test_deserializers_edge_cases.py` for edge cases

**Files to merge INTO `tests/coverage/test_deserializers_edge_cases.py`:**
- Functions from `tests/enhanced_types/test_numpy_auto_detection.py`
- Functions from `tests/enhanced_types/test_pandas_auto_detection.py`
- Functions from `tests/enhanced_types/test_basic_type_enhancements.py`
- Functions from `tests/coverage/test_lazy_imports_and_hotpath.py`
- Functions from `tests/features/test_template_deserialization.py` (edge cases only)

### Phase 3: Move Integration/Cross-Module Tests
**Target**: Clean separation of concerns

**Move to `tests/integration/`:**
- `tests/features/test_chunked_streaming.py` → `test_streaming_integration.py`
- `tests/features/test_ml_serializers.py` → `test_ml_integration.py`
- `tests/benchmarks/*` → `test_performance_integration.py`
- Standalone files → appropriate integration files

## Expected Result

### Final Structure (2-3 files per module max):
```
tests/
├── unit/                           # Main functionality (1 file per module)
│   ├── test_core_comprehensive.py
│   ├── test_deserializers_comprehensive.py
│   ├── test_api_comprehensive.py
│   ├── test_config_comprehensive.py
│   ├── test_serializers_comprehensive.py
│   ├── test_validation_comprehensive.py
│   ├── test_converters_comprehensive.py
│   └── test_data_utils_comprehensive.py
│
├── coverage/                       # Edge cases (1-2 files per module)
│   ├── test_core_edge_cases.py            # All core edge cases
│   ├── test_deserializers_edge_cases.py   # All deserializer edge cases
│   └── test_*.py                          # Other coverage files
│
└── integration/                    # Cross-module tests
    ├── test_streaming_integration.py
    ├── test_ml_integration.py
    ├── test_performance_integration.py
    └── test_*.py
```

## Benefits
- **Max 2-3 files per datason module**
- **Clear purpose**: unit/ = main, coverage/ = edge cases
- **Easy to find**: Know exactly where each test type belongs
- **No duplication**: Each test function in exactly one place
