# DataSON External Benchmark Integration - Setup Guide

## 🎯 **Overview**

This guide sets up automated PR performance testing using the **external `datason-benchmarks` repository**. Every DataSON PR will trigger comprehensive benchmarks in your separate benchmark repository.

**Architecture**: DataSON PR → Build Wheel → Trigger External Repo → Run Benchmarks → Post Results

---

## 🔑 **Step 1: Create GitHub Token**

### **Create Personal Access Token**
1. Go to: https://github.com/settings/tokens
2. Click **"Generate new token (classic)"**
3. Configure:
   ```
   Token Name: DataSON Benchmark Integration
   Expiration: 90 days

   Required Scopes:
   ☑️ repo (Full control of repositories)
   ☑️ workflow (Update GitHub Action workflows)
   ```
4. Click **"Generate token"** and **copy immediately**

### **Add Repository Secret**
1. Go to: https://github.com/danielendler/datason/settings/secrets/actions
2. Click **"New repository secret"**
3. Name: `BENCHMARK_REPO_TOKEN`
4. Value: [Your token from above]
5. Click **"Add secret"**

---

## 🏗️ **Step 2: Set Up External Repository**

### **Create `datason-benchmarks` Repository**
1. Create repository: https://github.com/new
2. Repository name: `datason-benchmarks`
3. Set as **Public** or **Private** (your choice)
4. Initialize with README

### **Add Required Files**
Follow the detailed setup in: [External Benchmark Setup Guide](EXTERNAL_BENCHMARK_SETUP.md)

**Key files needed:**
- `.github/workflows/datason-pr-integration.yml` (main workflow)
- `scripts/pr_optimized_benchmark.py` (benchmark script)
- `requirements.txt` (dependencies)

---

## 🧪 **Step 3: Test Integration**

### **Create Test PR**
```bash
# 1. Create test branch
git checkout -b test/external-benchmark
echo "# Test external benchmark integration" >> README.md
git add README.md && git commit -m "test: trigger external benchmark"
git push -u origin test/external-benchmark

# 2. Create PR
gh pr create --title "Test: External Benchmark" --body "Testing external benchmark integration"
```

### **Monitor Workflows**
1. **DataSON workflow**: https://github.com/danielendler/datason/actions
   - Should build wheel and trigger external repository
2. **Benchmark workflow**: https://github.com/danielendler/datason-benchmarks/actions
   - Should download wheel, run benchmarks, and post results

### **Expected Results**
✅ DataSON builds wheel artifact  
✅ External repository receives trigger  
✅ Benchmarks run successfully  
✅ Results posted back to DataSON PR  
✅ Artifacts uploaded for analysis  

---

## 🔧 **Step 4: Customize Benchmarks**

### **Benchmark Types**
The workflow supports different benchmark types:
- **`pr_optimized`** (default): Fast, PR-focused tests
- **`quick`**: Basic performance validation
- **`competitive`**: Full comparison with other libraries

### **Manual Trigger**
You can manually trigger benchmarks:
```bash
# Via GitHub CLI
gh workflow run pr-performance-check.yml -f pr_number=123 -f benchmark_type=competitive

# Or via GitHub web interface:
# Go to Actions → PR Performance Benchmark → Run workflow
```

---

## 🎯 **Success Checklist**

- [ ] `BENCHMARK_REPO_TOKEN` secret created and added
- [ ] `datason-benchmarks` repository created
- [ ] Required workflow files added to external repository
- [ ] Test PR created and workflows triggered successfully
- [ ] Benchmark results posted back to DataSON PR
- [ ] Artifacts uploaded and accessible

---

## 🆘 **Troubleshooting**

### **Common Issues**

**❌ "Permission denied"**
- Verify token has correct permissions (`repo` + `workflow`)
- Check token hasn't expired
- Ensure token has access to both repositories

**❌ "Workflow file not found"**
- Ensure file is exactly: `.github/workflows/datason-pr-integration.yml`
- Check it's on the `main` branch of `datason-benchmarks`

**❌ "Artifact download failed"**
- Verify artifact retention (7+ days)
- Check cross-repository artifact permissions

### **Debug Steps**
1. Check DataSON workflow logs for trigger success
2. Check external repository workflow logs for execution details
3. Verify all required inputs are passed correctly
4. Test with manual workflow dispatch first

---

## 📊 **Monitoring**

### **Regular Maintenance**
- **Weekly**: Review workflow success rates
- **Monthly**: Update benchmark thresholds
- **Per Release**: Update performance baselines

### **Performance Tracking**
The external repository will maintain:
- Performance history and trends
- Regression detection baselines
- Competitive analysis results

---

## 🚀 **You're Ready!**

This setup provides:
✅ **Automated performance testing** on every DataSON PR  
✅ **Clean separation** between code and benchmarks  
✅ **Professional reporting** with detailed analysis  
✅ **Scalable architecture** for benchmark expansion  

For detailed external repository setup, see: [External Benchmark Setup Guide](EXTERNAL_BENCHMARK_SETUP.md)
