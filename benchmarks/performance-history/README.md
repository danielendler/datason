# Performance History

This directory contains historical performance results for tracking datason performance over time.

## 📊 **Structure**

- `performance_YYYYMMDD_HHMMSS_GITHASH.json` - Timestamped performance results from CI
- `latest_baseline.json` - Current CI baseline for comparison
- `analysis/` - Performance analysis scripts and reports

## 🎯 **Usage**

**View Recent Performance:**
```bash
ls -la benchmarks/performance-history/performance_*.json | tail -5
```

**Analyze Performance Trends:**
```bash
cd benchmarks
python -c "
import json
import glob
from datetime import datetime

files = sorted(glob.glob('performance-history/performance_*.json'))
for file in files[-5:]:
    with open(file) as f:
        data = json.load(f)

    timestamp = data['metadata']['timestamp']
    print(f'{timestamp}: {file}')
"
```

**Generate Performance Report:**
```bash
cd benchmarks
python performance_analysis.py
```

## 📈 **Automatic Collection**

Performance results are automatically collected:
- **Every push to main** - Full performance suite
- **Weekly schedule** - Baseline updates  
- **Manual triggers** - On-demand testing

## 🔍 **Data Retention**

- **CI Artifacts**: 90 days
- **Git History**: Last 30 performance runs
- **Charts**: Generated on main branch changes

This ensures we can track performance trends without bloating the repository.
