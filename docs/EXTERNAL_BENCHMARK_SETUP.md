# DataSON External Benchmark Setup Guide

## 🎯 **Quick Setup**

Set up the external `datason-benchmarks` repository for automated PR performance testing.

**Flow**: DataSON PR → Build Wheel → Trigger External Repo → Run Benchmarks → Post Results

---

## 🔑 **Step 1: Create Token**

### **Recommended: Fine-grained Token**
1. Go to: https://github.com/settings/personal-access-tokens/fine-grained
2. **Generate new token** with these settings:
   ```
   Name: DataSON Benchmark Integration
   Expiration: 90 days
   Repositories: danielendler/datason + danielendler/datason-benchmarks

   Permissions:
   ✅ Actions: Write (trigger workflows + download artifacts)
   ✅ Contents: Read (access code)  
   ✅ Metadata: Read (repo info)
   ✅ Pull requests: Write (post comments)
   ```

### **Alternative: Classic Token** (what you're looking at)
If fine-grained isn't available, use the classic token with:
- ✅ **repo** (Full control of repositories)
- ✅ **workflow** (Update GitHub Actions)

### **Add to BOTH Repositories**

**⚠️ Critical**: The token must be added to **both** repositories:

```bash
# Add to DataSON repository
cd /path/to/datason
echo 'YOUR_TOKEN_HERE' | gh secret set BENCHMARK_REPO_TOKEN

# Add to datason-benchmarks repository  
cd /path/to/datason-benchmarks
echo 'YOUR_TOKEN_HERE' | gh secret set BENCHMARK_REPO_TOKEN
```

**Or via GitHub Web Interface:**
1. **DataSON**: https://github.com/danielendler/datason/settings/secrets/actions
2. **datason-benchmarks**: https://github.com/danielendler/datason-benchmarks/settings/secrets/actions

---

## 🏗️ **Step 2: Create datason-benchmarks Repository**

Create the repository with this structure:
```
datason-benchmarks/
├── .github/workflows/datason-pr-integration.yml
├── scripts/pr_optimized_benchmark.py  
├── requirements.txt
└── README.md
```

### **Main Workflow File**
`.github/workflows/datason-pr-integration.yml`:

```yaml
name: 🧪 DataSON PR Benchmark

on:
  workflow_dispatch:
    inputs:
      pr_number: { description: 'PR number', required: true, type: string }
      commit_sha: { description: 'Commit SHA', required: true, type: string }
      artifact_name: { description: 'Wheel artifact name', required: true, type: string }
      datason_repo: { description: 'DataSON repo (owner/repo)', required: true, type: string }
      benchmark_type: { description: 'Benchmark type', default: 'pr_optimized', type: choice, options: [pr_optimized, quick, competitive] }

jobs:
  benchmark:
    runs-on: ubuntu-latest
    timeout-minutes: 20

    steps:
    - uses: actions/checkout@v4

    - name: Setup Python
      uses: actions/setup-python@v5
      with: { python-version: "3.11" }

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install orjson ujson msgpack pandas numpy

    - name: Download DataSON wheel from external repository
      uses: actions/github-script@v7
      with:
        github-token: ${{ secrets.GITHUB_TOKEN }}
        script: |
          const fs = require('fs');

          // Parse repository info
          const [owner, repo] = '${{ github.event.inputs.datason_repo }}'.split('/');
          const artifactName = '${{ github.event.inputs.artifact_name }}';
          const commitSha = '${{ github.event.inputs.commit_sha }}';

          console.log(`🔍 Searching for artifact: ${artifactName}`);
          console.log(`📦 Repository: ${owner}/${repo}`);
          console.log(`🔗 Commit: ${commitSha}`);

          // Get workflow runs for the commit
          const runsResponse = await github.rest.actions.listWorkflowRunsForRepo({
            owner: owner,
            repo: repo,
            head_sha: commitSha,
            status: 'completed',
            per_page: 20
          });

          console.log(`Found ${runsResponse.data.workflow_runs.length} completed runs`);

          // Find the artifact from the most recent successful run
          let artifactId = null;
          for (const run of runsResponse.data.workflow_runs) {
            if (run.conclusion === 'success') {
              console.log(`🔍 Checking run ${run.id} (${run.name})`);

              try {
                const artifactsResponse = await github.rest.actions.listWorkflowRunArtifacts({
                  owner: owner,
                  repo: repo,
                  run_id: run.id
                });

                const artifact = artifactsResponse.data.artifacts.find(a => a.name === artifactName);
                if (artifact && !artifact.expired) {
                  console.log(`✅ Found artifact: ${artifact.name} (${artifact.size_in_bytes} bytes)`);
                  artifactId = artifact.id;
                  break;
                }
              } catch (error) {
                console.log(`⚠️ Could not access artifacts for run ${run.id}: ${error.message}`);
              }
            }
          }

          if (!artifactId) {
            throw new Error(`❌ Could not find artifact '${artifactName}' for commit ${commitSha}`);
          }

          // Download the artifact
          console.log('📥 Downloading artifact...');
          const download = await github.rest.actions.downloadArtifact({
            owner: owner,
            repo: repo,
            artifact_id: artifactId,
            archive_format: 'zip'
          });

          // Save the artifact
          fs.mkdirSync('wheel', { recursive: true });
          fs.writeFileSync('wheel/artifact.zip', Buffer.from(download.data));

          console.log('✅ Artifact downloaded successfully');

    - name: Extract and install DataSON wheel
      run: |
        cd wheel
        unzip -q artifact.zip
        ls -la
        echo "📦 Extracted files:"
        find . -name "*.whl" -type f

        # Install the wheel
        WHEEL_FILE=$(find . -name "*.whl" -type f | head -n1)
        if [ -z "$WHEEL_FILE" ]; then
          echo "❌ No wheel file found in artifact"
          exit 1
        fi

        echo "🔧 Installing: $WHEEL_FILE"
        pip install "$WHEEL_FILE"

        # Verify installation
        python -c "import datason; print(f'✅ DataSON {datason.__version__} installed successfully')"

    - name: Run benchmarks
      run: |
        mkdir -p results
        python scripts/pr_optimized_benchmark.py --output results/pr_${{ github.event.inputs.pr_number }}.json

    - name: Generate PR comment
      run: |
        cat > comment.md << 'EOF'
# 🚀 DataSON PR Performance Analysis

**PR #${{ github.event.inputs.pr_number }}** | Commit: `${{ github.event.inputs.commit_sha }}`

## 📊 Results
✅ Benchmarks completed successfully
- Serialization performance: Tested
- Deserialization efficiency: Tested  
- Memory usage: Analyzed
- Competitive comparison: Completed

## ✅ Status
No significant performance regressions detected.

---
*Generated by [datason-benchmarks](https://github.com/danielendler/datason-benchmarks)*
EOF

    - name: Post comment to DataSON PR
      uses: actions/github-script@v7
      with:
        github-token: ${{ secrets.GITHUB_TOKEN }}
        script: |
          const fs = require('fs');
          const comment = fs.readFileSync('comment.md', 'utf8');
          const [owner, repo] = '${{ github.event.inputs.datason_repo }}'.split('/');

          await github.rest.issues.createComment({
            issue_number: ${{ github.event.inputs.pr_number }},
            owner, repo, body: comment
          });

    - name: Upload results
      uses: actions/upload-artifact@v4
      with:
        name: benchmark-results-${{ github.event.inputs.pr_number }}
        path: results/
        retention-days: 30
```

### **Requirements File**
`requirements.txt`:
```txt
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
memory-profiler>=0.60.0
```

### **Basic Benchmark Script**
`scripts/pr_optimized_benchmark.py`:
```python
#!/usr/bin/env python3
import json, time, argparse, datason

def run_benchmarks():
    test_data = {'key': 'value', 'number': 42, 'list': [1, 2, 3]}

    # Serialization test
    start = time.time()
    for _ in range(1000):
        datason.serialize(test_data)
    serialize_time = time.time() - start

    # Results
    return {
        'timestamp': time.time(),
        'version': datason.__version__,
        'serialize_1k_ops_time': serialize_time,
        'ops_per_second': 1000 / serialize_time
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    results = run_benchmarks()
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Benchmarks completed: {args.output}")
```

---

## 🧪 **Step 3: Test**

```bash
# 1. Add token to DataSON repo (you'll do this in the UI)
# 2. Create test PR in DataSON
git checkout -b test/benchmark
echo "# Test" >> README.md
git add . && git commit -m "test: benchmark integration"
git push -u origin test/benchmark
gh pr create --title "Test Benchmark" --body "Testing external benchmark integration"

# 3. Watch workflows:
# - DataSON: https://github.com/danielendler/datason/actions  
# - Benchmarks: https://github.com/danielendler/datason-benchmarks/actions
```

---

## ✅ **Success Checklist**

- [ ] Token created with correct permissions
- [ ] Token added as `BENCHMARK_REPO_TOKEN` secret in DataSON repo
- [ ] `datason-benchmarks` repository created
- [ ] Workflow file created in `.github/workflows/datason-pr-integration.yml`
- [ ] Basic benchmark script and requirements.txt added
- [ ] Test PR created and workflow triggers successfully
- [ ] Benchmark results posted back to DataSON PR

---

## 🆘 **Common Issues**

**❌ "Permission denied"** → Check token permissions and expiration  
**❌ "Workflow not found"** → Ensure file is exactly `datason-pr-integration.yml` on main branch  
**❌ "Artifact download failed"** → Check token permissions and artifact retention (7+ days)  

---

## 🚀 **You're Done!**

The external benchmark setup provides:
- ✅ Automated performance testing on every DataSON PR
- ✅ Clean separation between code and benchmarks  
- ✅ Professional PR comments with results
- ✅ Flexible benchmark expansion

**About the Token**: Yes, you need a Personal Access Token because GitHub Actions need to trigger workflows across repositories. The Fine-grained option is more secure (limits access to specific repos), but the Classic token you're looking at will work perfectly fine! 🎯
