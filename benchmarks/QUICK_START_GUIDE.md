# Quick Start Guide: Performance Analysis After Each Improvement

## TL;DR - After Each Code Change

```bash
# Quick check (30 seconds)
cd benchmarks
python run_performance_analysis.py --quick --compare

# Full analysis (2-3 minutes)
python run_performance_analysis.py --compare

# Save as new baseline after confirmed improvement
python run_performance_analysis.py --save-baseline
```

## Setup (One-Time)

### 1. Install Benchmarking Dependencies
```bash
cd benchmarks
pip install -r requirements-benchmarking.txt
```

This installs:
- **NumPy, Pandas** (for realistic ML testing)
- **OrJSON, UJSON, msgpack** (for competitive analysis)
- **Performance profiling tools** (optional)

### 2. Establish Baseline
```bash
python run_performance_analysis.py --save-baseline
```

## Daily Workflow

### After Each Small Improvement

**1. Quick Check** (30 seconds):
```bash
python run_performance_analysis.py --quick --compare
```

**Output Example**:
```
🎯 Datason Performance Analysis
   Version: 0.4.5
   Commit:  a1b2c3d4
   Branch:  feature/type-caching

🟢 Performance Improvements (2)
  serialization.api_response.standard: -8.3% (1.45ms ← 1.58ms)
  type_detection.mixed_list: -12.1% (0.89ms ← 1.01ms)

💡 Quick analysis shows 8-12% improvements ✅
```

**2. Full Analysis** (2-3 minutes) if quick check shows improvement:
```bash  
python run_performance_analysis.py --compare
```

**Output includes competitive analysis**:
```
📊 Competitive Position:
  vs json_standard: 7.2x slower (was 7.8x - improved!)
  vs orjson: 54.1x slower (was 58.2x - improved!)

💡 Key Recommendations:
  🔥 CRITICAL: Still 54x slower than OrJSON
  ✅ IMPROVEMENT: 6% better competitive position
```

### When to Save Baseline

**Save baseline** when you have confirmed improvements:
```bash
python run_performance_analysis.py --save-baseline
```

**Don't save baseline** if:
- Performance regressed
- Changes were experimental  
- Working on work-in-progress branch

## Understanding the Output

### Version Tracking
```
🎯 Datason Performance Analysis
   Version: 0.4.5              # datason package version
   Commit:  a1b2c3d4            # Git commit hash
   Branch:  feature/type-caching # Current branch
   ⚠️  WARNING: Uncommitted changes detected!
```

### Performance Changes
```
🟢 Performance Improvements (2)
  test_name: -8.3% (1.45ms ← 1.58ms)
  # Negative % = faster (good)
  # Shows: current_time ← previous_time

🔴 Performance Regressions (1)  
  test_name: +15.2% (2.34ms ← 2.03ms)
  # Positive % = slower (bad)
```

### Competitive Position
```
📊 Competitive Position:
  vs json_standard: 7.2x slower    # vs Python's json module
  vs orjson: 54.1x slower          # vs fastest JSON library
  vs pickle: 18.3x slower          # vs Python pickle
```

## Different Analysis Types

### `--quick` (30 seconds)
- Core functionality only
- No ML libraries needed
- Good for rapid iteration
- CI-style tests

### Default/Full (2-3 minutes)
- ML library integration
- Competitive analysis  
- Real-world data complexity
- Complete picture

### `--competitive`
- Forces competitive analysis
- Requires orjson, ujson, msgpack
- Use when evaluating major changes

## File Organization

Results are saved with version tracking:
```
benchmarks/results/
├── performance_comprehensive_0.4.5_a1b2c3d4_20250602_143022.json
├── performance_quick_0.4.5_a1b2c3d4_20250602_143155.json  
├── baseline.json                    # Current baseline for CI
└── latest.json                      # Most recent results
```

Filename format: `performance_{type}_{version}_{commit}_{timestamp}.json`

## Typical Development Cycle

### 1. Before Starting Work
```bash
# Establish baseline for your branch
python run_performance_analysis.py --save-baseline
```

### 2. After Each Small Change
```bash
# Quick check - did it help?
python run_performance_analysis.py --quick --compare
```

### 3. After Significant Improvement
```bash
# Full analysis with competitive position
python run_performance_analysis.py --compare

# If results are good, save new baseline
python run_performance_analysis.py --save-baseline
```

### 4. Before Creating PR
```bash
# Final comprehensive analysis
python run_performance_analysis.py --competitive --compare
```

## Troubleshooting

### Missing Dependencies
```
⚠️  Missing dependencies: numpy, pandas
   Install with: pip install -r requirements-benchmarking.txt
```

**Solution**: `pip install -r requirements-benchmarking.txt`

### No Previous Results
```
⚠️  No previous results found for comparison
```

**Solution**: This is normal for first run. Results will be available for next comparison.

### Competitive Libraries Missing
```
💡 Competitive libraries not available: orjson, ujson
```

**Solution**: Install with `pip install orjson ujson msgpack`

## Best Practices

### ✅ **Do**
- Run quick analysis after each small change
- Save baseline only after confirmed improvements
- Check version info to ensure you're testing the right code
- Use `--compare` to track progress over time

### ❌ **Don't**  
- Save baseline after regressions
- Ignore uncommitted changes warnings
- Skip competitive analysis for major changes
- Compare results from different environments

## Integration with CI

The system integrates with GitHub Actions:
- **Daily CI**: Uses `ci_performance_tracker.py` (Tier 1)
- **Monthly CI**: Uses `comprehensive_performance_suite.py` (Tier 2+3)
- **Manual**: Use `run_performance_analysis.py` for on-demand testing

## Example Session

```bash
# Start work on optimization
cd benchmarks
python run_performance_analysis.py --save-baseline

# ... make some changes ...

# Quick check
python run_performance_analysis.py --quick --compare
# ✅ Shows 8% improvement

# Full analysis  
python run_performance_analysis.py --compare
# ✅ Shows competitive position improved

# Save progress
python run_performance_analysis.py --save-baseline

# ... continue iterating ...
```

This gives you **scientific measurement** of each improvement with **full version tracking** and **competitive context**. Perfect for making data-driven optimization decisions! 🚀
