"""Tests for error handling and security limits."""

import pytest

import datason
from datason import SecurityError
from datason.config import SerializationConfig


class TestErrorHandling:
    """Test error handling and security features."""

    def test_basic_serialization(self) -> None:
        """Test that basic serialization works."""
        config = SerializationConfig()
        result = datason.serialize(123, config=config)
        assert result == 123

    def test_security_limits(self) -> None:
        """Test that security limits (e.g., max depth) are enforced.

        DEBUGGING HISTORY AND COMPREHENSIVE NOTES:
        ==========================================

        ORIGINAL PROBLEM:
        - CI was failing with: "Expected SecurityError or RecursionError but got SecurityError: Maximum serialization depth (3) exceeded. Current depth: 4"
        - This bizarre error suggested pytest was saying we got SecurityError when we expected SecurityError
        - Test passed individually but failed when run with other tests (test pollution issue)
        - Issue appeared specifically with --maxfail=1 flag in CI environment

        FAILED ATTEMPTS AND WHY THEY DIDN'T WORK:
        ----------------------------------------

        1. INITIAL SOLUTION: Simple except clause
           ```python
           except (SecurityError, RecursionError) as e:
           ```
           PROBLEM: Failed in CI with the same weird error message
           ROOT CAUSE: Exception class identity issues between imports

        2. BACKUP TEST ANALYSIS: Used explicit exception handling
           ```python
           try:
               result = serialize(...)
               pytest.fail("Expected exception")
           except (SecurityError, RecursionError) as e:
               # handle success
           except Exception as e:
               pytest.fail(f"Got {type(e).__name__}: {e}")
           ```
           PROBLEM: Still failed because SecurityError wasn't caught by the first except clause
           ROOT CAUSE: The SecurityError being raised had different class identity than imported SecurityError

        3. TEST ISOLATION ATTEMPTS:
           - Added datason.clear_caches() and datason.reset_default_config()
           - Added garbage collection
           - Added finally blocks for cleanup
           PROBLEM: Didn't fix the core exception identity issue
           OUTCOME: Good practice but didn't solve the root problem

        4. RESEARCH FINDINGS:
           - This is a known pytest issue with exception class identity
           - Can happen when modules are imported differently across tests
           - Common in complex test environments with module reloading
           - Similar to Python ExceptionGroup handling issues in pytest

        FINAL WORKING SOLUTION:
        ----------------------

        DUAL EXCEPTION TYPE CHECKING:
        ```python
        except Exception as e:
            actual_exception_type = type(e).__name__
            if actual_exception_type in ('SecurityError', 'RecursionError') or isinstance(e, (SecurityError, RecursionError)):
                # Handle as expected exception
            else:
                pytest.fail(f"Got unexpected {actual_exception_type}: {e}")
        ```

        WHY THIS WORKS:
        - String-based type checking (type(e).__name__) works even with class identity issues
        - isinstance() checking works for normal cases
        - Catches ALL exceptions first, then filters by type
        - More robust against module import ordering issues
        - Works in both local and CI environments

        LESSON LEARNED:
        - Always use dual exception type checking in complex test environments
        - Don't rely solely on exception class identity in pytest
        - Test isolation is important but doesn't fix class identity issues
        - CI environments can have different module loading behavior than local

        FUTURE DEBUGGING:
        - If this test fails again, check if SecurityError is being imported differently
        - Verify that the exception message contains depth-related keywords
        - Consider using exception.args or str(exception) for more robust checking
        """
        # Clear caches and reset global state for clean isolation
        # NOTE: This prevents test pollution but doesn't fix the exception identity issue
        datason.clear_caches()
        datason.reset_default_config()

        # Import garbage collection for thorough cleanup
        import gc

        gc.collect()

        try:
            # Test max depth with explicit config - create deeper nesting to ensure limit is hit
            nested_data = {"level1": {"level2": {"level3": {"level4": {"level5": {"level6": "too_deep"}}}}}}
            config_depth_3 = SerializationConfig(max_depth=3)

            # Test that deep nesting raises security-related exception
            # Based on backup tests, this might be SecurityError or RecursionError
            security_exception_raised = False
            exception_message = ""
            actual_exception_type = None

            try:
                result = datason.serialize(nested_data, config=config_depth_3)
                # If no exception, something is wrong
                pytest.fail(f"Expected security exception but serialization succeeded: {result}")
            except Exception as e:
                # CRITICAL: Catch ALL exceptions first, then filter by type
                # This approach works around pytest exception class identity issues
                actual_exception_type = type(e).__name__
                exception_message = str(e)

                # DUAL EXCEPTION TYPE CHECKING:
                # 1. String-based checking works even with class identity issues
                # 2. isinstance() checking works for normal cases
                # This combination makes the test robust in all environments
                if actual_exception_type in ("SecurityError", "RecursionError") or isinstance(
                    e, (SecurityError, RecursionError)
                ):
                    security_exception_raised = True
                    # Verify the exception message mentions depth/security concepts
                    assert any(
                        word in exception_message.lower()
                        for word in ["depth", "maximum", "limit", "security", "recursion"]
                    )
                else:
                    # This should never happen now, but provides clear debugging info if it does
                    pytest.fail(f"Expected SecurityError or RecursionError but got {actual_exception_type}: {e}")

            # Final verification that we got the expected exception
            assert security_exception_raised, (
                f"Expected security exception to be raised for deep nesting. Got type: {actual_exception_type}, message: {exception_message}"
            )

            # Test success case: no error if within limits
            config_depth_10 = SerializationConfig(max_depth=10)
            result = datason.serialize(nested_data, config=config_depth_10)
            assert result is not None  # Should succeed with higher limit

        finally:
            # Always restore defaults to avoid test pollution
            # This is good practice even though it doesn't fix the main issue
            datason.clear_caches()
            datason.reset_default_config()
            gc.collect()

    def test_large_object_limits(self) -> None:
        """Test that limits on large objects (e.g., strings) are handled by truncation."""
        config = SerializationConfig(max_string_length=10)

        # Test string length limit - truncates and adds [TRUNCATED] marker
        result = datason.serialize("a" * 15, config=config)
        assert "[TRUNCATED]" in result

        # Test no truncation if within limits
        result = datason.serialize("a" * 10, config=config)
        assert result == "a" * 10
