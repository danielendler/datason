#!/usr/bin/env python3
"""
Debug script for DataSON benchmark integration token issues.

This script helps diagnose GitHub token permission problems when triggering
external benchmark workflows.
"""

import os
import sys
from datetime import datetime

import requests


def check_token_permissions() -> bool:
    """Check if the GitHub token has the required permissions."""

    # Get token from environment (for local testing)
    token = os.getenv("GITHUB_TOKEN") or os.getenv("BENCHMARK_REPO_TOKEN")

    if not token:
        print("❌ No GitHub token found in environment variables")
        print("   Set GITHUB_TOKEN or BENCHMARK_REPO_TOKEN environment variable")
        print("   Example: export GITHUB_TOKEN='your_token_here'")
        return False

    print("🔍 Testing token permissions...")
    print(f"   Token: {token[:8]}...{token[-4:]} (masked)")

    # Test basic token validity
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "DataSON-Debug-Script",
    }

    try:
        # Check user info and token scopes
        response = requests.get("https://api.github.com/user", headers=headers)

        if response.status_code != 200:
            print(f"❌ Token authentication failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

        user_data = response.json()
        scopes = response.headers.get("X-OAuth-Scopes", "").split(", ")

        print(f"✅ Token authenticated as: {user_data.get('login')}")
        print(f"📋 Token scopes: {', '.join(scopes) if scopes != [''] else 'None'}")

        # Check required scopes
        required_scopes = ["repo", "workflow"]
        missing_scopes = []

        for scope in required_scopes:
            if scope not in scopes:
                missing_scopes.append(scope)

        if missing_scopes:
            print(f"❌ Missing required scopes: {', '.join(missing_scopes)}")
            print("   Required scopes for cross-repo workflow triggering:")
            print("   • repo (full repository access)")
            print("   • workflow (workflow management)")
            return False
        else:
            print("✅ Token has all required scopes")

        # Test access to datason repository
        print("\n🔍 Testing DataSON repository access...")
        repo_response = requests.get("https://api.github.com/repos/danielendler/datason", headers=headers)

        if repo_response.status_code == 200:
            print("✅ Can access danielendler/datason repository")
        else:
            print(f"❌ Cannot access danielendler/datason: {repo_response.status_code}")

        # Test access to datason-benchmarks repository
        print("\n🔍 Testing datason-benchmarks repository access...")
        bench_response = requests.get("https://api.github.com/repos/danielendler/datason-benchmarks", headers=headers)

        if bench_response.status_code == 200:
            print("✅ Can access danielendler/datason-benchmarks repository")

            # Check if workflow file exists
            workflow_response = requests.get(
                "https://api.github.com/repos/danielendler/datason-benchmarks/contents/.github/workflows/datason-pr-integration.yml",
                headers=headers,
            )

            if workflow_response.status_code == 200:
                print("✅ datason-pr-integration.yml workflow file exists")
            else:
                print("❌ datason-pr-integration.yml workflow file not found")
                print("   This workflow file must exist in the datason-benchmarks repository")

        elif bench_response.status_code == 404:
            print("❌ datason-benchmarks repository not found or not accessible")
            print("   Make sure the repository exists and the token has access")
        else:
            print(f"❌ Cannot access datason-benchmarks: {bench_response.status_code}")

        # Test workflow dispatch endpoint (dry run)
        print("\n🔍 Testing workflow dispatch endpoint...")
        dispatch_url = "https://api.github.com/repos/danielendler/datason-benchmarks/actions/workflows/datason-pr-integration.yml/dispatches"

        # Don't actually trigger, just test if we can access the endpoint
        test_headers = headers.copy()
        test_headers["Content-Type"] = "application/json"

        # Use a HEAD request to test endpoint accessibility
        head_response = requests.head(dispatch_url, headers=test_headers)

        if head_response.status_code == 405:  # Method not allowed is expected for HEAD
            print("✅ Workflow dispatch endpoint is accessible")
        elif head_response.status_code == 404:
            print("❌ Workflow dispatch endpoint not found")
            print("   Check that datason-pr-integration.yml exists and is on main branch")
        else:
            print(f"⚠️  Workflow dispatch endpoint returned: {head_response.status_code}")

        return True

    except requests.RequestException as e:
        print(f"❌ Network error: {e}")
        return False


def print_setup_instructions() -> None:
    """Print setup instructions for fixing token issues."""

    print("\n" + "=" * 60)
    print("🔧 TOKEN SETUP INSTRUCTIONS")
    print("=" * 60)

    print("\n1. CREATE NEW PERSONAL ACCESS TOKEN:")
    print("   • Go to: https://github.com/settings/tokens")
    print("   • Click 'Generate new token (classic)'")
    print("   • Select these scopes:")
    print("     ☑️ repo (Full control of repositories)")
    print("     ☑️ workflow (Update GitHub Action workflows)")
    print("     ☑️ actions:read (Download artifacts)")

    print("\n2. ADD TO REPOSITORY SECRETS:")
    print("   DataSON repository:")
    print("   • https://github.com/danielendler/datason/settings/secrets/actions")
    print("   • Name: BENCHMARK_REPO_TOKEN")
    print("   • Value: [your new token]")

    print("\n   datason-benchmarks repository:")
    print("   • https://github.com/danielendler/datason-benchmarks/settings/secrets/actions")
    print("   • Name: BENCHMARK_REPO_TOKEN")
    print("   • Value: [same token]")

    print("\n3. VERIFY WORKFLOW FILE EXISTS:")
    print("   • File: .github/workflows/datason-pr-integration.yml")
    print("   • Repository: danielendler/datason-benchmarks")
    print("   • Branch: main")

    print("\n4. TEST AGAIN:")
    print("   • Create a test PR in DataSON repository")
    print("   • Check workflow logs for detailed error messages")


def main() -> None:
    """Main debugging function."""

    print("🔧 DataSON Benchmark Integration Token Debugger")
    print("=" * 50)
    print(f"🕒 Timestamp: {datetime.now().isoformat()}")

    success = check_token_permissions()

    if not success:
        print_setup_instructions()
        sys.exit(1)
    else:
        print("\n✅ Token permissions look good!")
        print("   If you're still getting errors, check:")
        print("   • Workflow file exists in datason-benchmarks repository")
        print("   • Token is correctly set in repository secrets")
        print("   • Repository visibility settings allow cross-repo access")


if __name__ == "__main__":
    main()
