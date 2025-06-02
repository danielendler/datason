# Essential Files for Performance Tracking System

## Core Performance Analysis Scripts (REQUIRED)
```
benchmarks/ci_performance_tracker.py               # Tier 1: Daily CI regression detection
benchmarks/comprehensive_performance_suite.py      # Tier 2+3: ML + competitive analysis  
benchmarks/run_performance_analysis.py             # On-demand analysis after each change
```

## Configuration Files (REQUIRED)
```
benchmarks/requirements-benchmarking.txt           # Competitive libraries (separate from dev deps)
```

## CI Workflows (REQUIRED)
```
.github/workflows/performance.yml                  # Daily regression testing
.github/workflows/comprehensive-performance.yml    # Monthly comprehensive analysis
```

## Documentation (RECOMMENDED)
```
benchmarks/QUICK_START_GUIDE.md                   # How to use after each change
benchmarks/COMPLETE_SYSTEM_SUMMARY.md             # Overview of entire system
benchmarks/INCREMENTAL_PERFORMANCE_PLAN.md        # 4-phase optimization roadmap
```

## Implementation Helpers (OPTIONAL)
```
benchmarks/implement_step_1_1.py                  # Step-by-step guidance
```

## Analysis Documentation (INFORMATIONAL)
```
benchmarks/MULTI_TIER_BENCHMARKING_STRATEGY.md    # Why we need 3 tiers
benchmarks/PERFORMANCE_ANALYSIS_FINDINGS.md       # Current competitive analysis
```

## Results/Data Files (DO NOT COMMIT)
```
benchmarks/comprehensive_performance_*.json        # Generated results
benchmarks/results/                                # Result storage directory
```

## Files to EXCLUDE/CLEAN UP
```
benchmarks/benchmarks/                             # Duplicate directory
benchmarks/PERFORMANCE_DEEP_DIVE_ANALYSIS.md      # Old investigation (already staged)
benchmarks/performance_investigation_*.json        # Old investigation data
benchmarks/realistic_performance_investigation.py  # Old investigation script
benchmarks/simple_realistic_benchmarks.py         # Old investigation script
benchmarks/PERFORMANCE_SUMMARY.md                 # Redundant with COMPLETE_SYSTEM_SUMMARY.md
benchmarks/PERFORMANCE_TRACKING_README.md         # Redundant with QUICK_START_GUIDE.md
```
