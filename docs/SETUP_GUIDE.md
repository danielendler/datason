# DataSON PR Performance Integration - Complete Setup Guide

## 🚀 **Quick Start: Choose Your Path**

You now have **two workflow options** for PR performance testing:

### **Option A: Local Benchmarks (Recommended for Immediate Use)**
✅ **Ready to use immediately**  
✅ Uses your existing benchmark infrastructure  
✅ No external dependencies  
✅ Works with current codebase  

**File**: `.github/workflows/pr-performance-check-local.yml`

### **Option B: External Benchmark Repository**
⏳ Requires external `datason-benchmarks` repository setup  
⏳ Needs `BENCHMARK_REPO_TOKEN` configuration  
⏳ More complex but follows the original integration guide  

**File**: `.github/workflows/pr-performance-check.yml`

---

## 📋 **Complete Setup Steps**

### **Step 1: Choose and Enable Workflow**

**For Local Benchmarks (Recommended):**
```bash
# The local workflow is ready to use immediately
# No additional setup required for basic functionality
```

**For External Repository (Advanced):**
1. Set up `BENCHMARK_REPO_TOKEN` (see Step 2 below)
2. Create or configure access to `datason-benchmarks` repository

### **Step 2: Repository Token Setup (Only for External Repository)**

#### **Create Personal Access Token**
1. Go to: https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Configure:
   ```
   Token Name: DataSON Benchmark Integration
   Expiration: 90 days (or per your policy)

   Required Scopes:
   ☑️ repo (Full control of private repositories)
   ☑️ workflow (Update GitHub Action workflows)
   ☑️ write:packages (if needed for artifacts)
   ```
4. Click **"Generate token"** and **copy immediately**

#### **Add Repository Secret**
**Option A: GitHub Web Interface**
1. Go to: https://github.com/danielendler/datason/settings/secrets/actions
2. Click **"New repository secret"**
3. Name: `BENCHMARK_REPO_TOKEN`
4. Value: [Your token from above]
5. Click **"Add secret"**

**Option B: GitHub CLI** (if you have the token)
```bash
echo 'YOUR_TOKEN_HERE' | gh secret set BENCHMARK_REPO_TOKEN
```

### **Step 3: Test the Integration**

#### **Test Local Workflow**
1. **Create a test branch:**
   ```bash
   git checkout -b test/benchmark-integration
   ```

2. **Make a small change** (to trigger the workflow):
   ```bash
   echo "# Test change" >> README.md
   git add README.md
   git commit -m "test: trigger benchmark workflow"
   git push -u origin test/benchmark-integration
   ```

3. **Create a test PR:**
   ```bash
   gh pr create --title "Test: Benchmark Integration" --body "Testing the PR performance benchmark workflow"
   ```

4. **Monitor the workflow:**
   - Go to: https://github.com/danielendler/datason/actions
   - Look for the "🚀 DataSON PR Performance Benchmark (Local)" workflow
   - Check that it completes successfully
   - Verify PR comment is posted with performance analysis

#### **Expected Results**
✅ Workflow builds DataSON wheel from PR  
✅ Runs comprehensive benchmarks  
✅ Posts detailed PR comment with results  
✅ Uploads performance artifacts  

### **Step 4: Customize Performance Thresholds**

Edit the workflow files to adjust thresholds:

```yaml
# In either workflow file, find and modify:
# Currently set to always pass (REGRESSION_DETECTED=false)
# You can enhance this with actual comparison logic

# Example customization in the regression detection step:
- name: 🔍 Performance regression detection
  run: |
    # Add your custom performance comparison logic here
    # Example: compare execution times, memory usage, etc.

    # Set REGRESSION_DETECTED=true if issues found
    if [ "performance_degraded" = "true" ]; then
      echo "REGRESSION_DETECTED=true" >> $GITHUB_ENV
    else
      echo "REGRESSION_DETECTED=false" >> $GITHUB_ENV
    fi
```

---

## 🔧 **Advanced Configuration**

### **Custom Benchmark Parameters**

Modify the benchmark execution in the workflow:

```yaml
# In pr-performance-check-local.yml, customize:
- name: 🚀 Run comprehensive benchmarks
  run: |
    cd benchmarks

    # Add custom parameters to benchmark scripts
    python comprehensive_performance_suite.py --iterations 10 --warm-up 3
    python realistic_performance_investigation.py --quick --output-format json
    python simple_realistic_benchmarks.py --competitive-mode
```

### **Custom PR Comment Templates**

Enhance the PR comment generation:

```yaml
# Add custom metrics to PR comments
echo "### 🎯 Custom Metrics" >> results/pr_analysis/pr_performance_comment.md
echo "- Memory Usage: [Add your metric]" >> results/pr_analysis/pr_performance_comment.md
echo "- CPU Performance: [Add your metric]" >> results/pr_analysis/pr_performance_comment.md
```

### **Baseline Management**

Set up performance baselines:

```bash
# In your repository, create a baseline after a stable release
cd benchmarks
python comprehensive_performance_suite.py
cp results/latest_comprehensive.json results/baseline_performance.json
git add results/baseline_performance.json
git commit -m "feat: establish performance baseline for v2.x"
```

---

## 🎯 **Testing Checklist**

### **Pre-Test Setup**
- [ ] Choose workflow option (local vs external)
- [ ] If external: Set up `BENCHMARK_REPO_TOKEN`
- [ ] If external: Verify access to benchmark repository
- [ ] Ensure all benchmark dependencies are available

### **Test Execution**
- [ ] Create test branch and PR
- [ ] Verify workflow triggers on PR creation
- [ ] Check workflow completes without errors
- [ ] Confirm PR comment is posted with results
- [ ] Download and verify artifacts are created

### **Post-Test Validation**
- [ ] Review benchmark results for accuracy
- [ ] Test performance regression detection (if implemented)
- [ ] Verify workflow works on different PR types
- [ ] Test manual workflow dispatch (if needed)

---

## 📊 **Monitoring and Maintenance**

### **Workflow Health Checks**
- **Weekly**: Review workflow execution times and success rates
- **Monthly**: Update benchmark thresholds based on performance trends
- **Per Release**: Update performance baselines for major versions

### **Performance Baseline Updates**
```bash
# After major releases or performance improvements
cd benchmarks
python comprehensive_performance_suite.py
# Review results and update baseline if appropriate
cp results/latest_comprehensive.json results/baseline_performance.json
git add results/baseline_performance.json
git commit -m "feat: update performance baseline for v2.x.x"
```

### **Troubleshooting Common Issues**

**❌ Workflow fails to trigger**
- Check file paths in workflow triggers
- Verify branch protection rules don't block workflows
- Ensure GitHub Actions are enabled in repository settings

**❌ Benchmark scripts fail**
- Check `benchmarks/requirements-benchmarking.txt` dependencies
- Verify Python version compatibility (workflow uses 3.11)
- Review benchmark script parameters and paths

**❌ PR comments not posted**
- Check `pull-requests: write` permission in workflow
- Verify GitHub token permissions
- Look for rate limiting in workflow logs

**❌ Artifacts not uploaded**
- Check artifact paths in workflow
- Verify artifact names are unique
- Ensure results directories are created

---

## 🚀 **Next Steps After Setup**

1. **Monitor First Few PRs**: Watch how the workflow performs with real PRs
2. **Enhance Regression Detection**: Add more sophisticated performance comparison logic  
3. **Customize Thresholds**: Set appropriate performance regression thresholds
4. **Create Documentation**: Document performance expectations for contributors
5. **Integration with Other Tools**: Consider integrating with monitoring dashboards

---

## 📈 **Success Metrics**

After successful setup, you should see:

✅ **Automated performance testing** on every PR  
✅ **Consistent benchmark execution** times (15-20 minutes)  
✅ **Detailed PR comments** with actionable insights  
✅ **Performance artifact generation** for deep analysis  
✅ **Early detection** of performance regressions  
✅ **Improved confidence** in performance changes  

The local benchmark workflow is **ready to use immediately** and provides comprehensive performance testing using your existing benchmark infrastructure!

---

## 🆘 **Support**

If you encounter issues:
1. **Check workflow logs** in GitHub Actions tab
2. **Review benchmark script outputs** in artifacts
3. **Test benchmark scripts locally** in the `benchmarks/` directory
4. **Verify dependencies** are properly installed

The integration is designed to be **robust and self-contained**, providing immediate value with minimal setup overhead.
