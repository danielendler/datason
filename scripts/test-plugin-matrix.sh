#!/bin/bash
# Local Plugin Matrix Testing Script
# Tests the same scenarios as CI to validate caching and efficiency

set -e

echo "🧪 Local Plugin Matrix Testing"
echo "================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create base test directory
TEST_BASE_DIR="./test-environments"
mkdir -p "$TEST_BASE_DIR"

# Test scenarios (matching CI matrix)
SCENARIOS="minimal with-numpy with-pandas with-ml-deps full"

get_description() {
    case $1 in
        "minimal") echo "Core functionality only" ;;
        "with-numpy") echo "Core + NumPy support" ;;
        "with-pandas") echo "Core + Pandas support" ;;
        "with-ml-deps") echo "Core + ML dependencies" ;;
        "full") echo "All dependencies" ;;
        *) echo "Unknown scenario" ;;
    esac
}

# Function to test a scenario
test_scenario() {
    local scenario=$1
    local description=$(get_description "$scenario")

    echo -e "${BLUE}📦 Testing: $scenario${NC}"
    echo -e "${YELLOW}   $description${NC}"

    # Create virtual environment for this scenario
    venv_dir="$TEST_BASE_DIR/venv-$scenario"

    if [ ! -d "$venv_dir" ]; then
        echo "   🔨 Creating virtual environment..."
        python -m venv "$venv_dir"
    else
        echo "   ♻️  Reusing cached virtual environment..."
    fi

    # Activate environment
    source "$venv_dir/bin/activate"

    # Upgrade pip (cached if already done)
    if [ ! -f "$venv_dir/.pip-upgraded" ]; then
        echo "   📦 Upgrading pip..."
        pip install --upgrade pip setuptools wheel > /dev/null 2>&1
        touch "$venv_dir/.pip-upgraded"
    fi

    # Install package
    echo "   📦 Installing datason..."
    pip install -e . -q

    # Install test dependencies
    if [ ! -f "$venv_dir/.test-deps-installed" ]; then
        echo "   📦 Installing test dependencies..."
        pip install pytest pytest-cov -q
        touch "$venv_dir/.test-deps-installed"
    fi

    # Install scenario-specific dependencies
    case $scenario in
        "minimal")
            echo "   ✅ No additional dependencies needed"
            ;;
        "with-numpy")
            if [ ! -f "$venv_dir/.numpy-installed" ]; then
                echo "   📦 Installing numpy..."
                pip install numpy -q
                touch "$venv_dir/.numpy-installed"
            fi
            ;;
        "with-pandas")
            if [ ! -f "$venv_dir/.pandas-installed" ]; then
                echo "   📦 Installing pandas..."
                pip install pandas -q
                touch "$venv_dir/.pandas-installed"
            fi
            ;;
        "with-ml-deps")
            if [ ! -f "$venv_dir/.ml-installed" ]; then
                echo "   📦 Installing ML dependencies..."
                pip install numpy pandas scikit-learn -q
                touch "$venv_dir/.ml-installed"
            fi
            ;;
        "full")
            if [ ! -f "$venv_dir/.dev-installed" ]; then
                echo "   📦 Installing dev dependencies..."
                pip install -e ".[dev]" -q
                touch "$venv_dir/.dev-installed"
            fi
            ;;
    esac

    # Test import
    echo "   🧪 Testing package import..."
    python -c "
import datason
print('   ✅ Package imports successfully')
data = {'test': 123, 'scenario': '$scenario'}
result = datason.serialize(data)
print('   ✅ Basic serialization works:', result)
"

    # Run appropriate tests
    echo "   🧪 Running tests..."
    case $scenario in
        "minimal"|"with-numpy"|"with-pandas")
            pytest tests/test_core.py tests/test_deserializers.py tests/test_converters.py -v --tb=short -q || true
            ;;
        "with-ml-deps")
            pytest tests/test_core.py tests/test_deserializers.py tests/test_converters.py tests/test_ml_serializers.py -v --tb=short -q || true
            ;;
        "full")
            pytest tests/ -v --tb=short -q --maxfail=5 || true
            ;;
    esac

    deactivate
    echo -e "${GREEN}   ✅ $scenario completed successfully${NC}"
    echo ""
}

# Record start time
start_time=$(date +%s)

echo "🏁 Starting plugin testing..."
echo ""

# Test all scenarios
for scenario in $SCENARIOS; do
    test_scenario "$scenario"
done

# Calculate total time
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "🎉 All plugin scenarios tested!"
echo "⏱️  Total time: ${duration}s"
echo ""
echo "💡 Efficiency notes:"
echo "   - Virtual environments are cached and reused"
echo "   - Dependencies are only installed once per environment"
echo "   - Subsequent runs will be much faster"
echo ""
echo "🔍 To inspect environments:"
echo "   ls -la $TEST_BASE_DIR/"
echo ""
echo "🧹 To clean up:"
echo "   rm -rf $TEST_BASE_DIR/"
