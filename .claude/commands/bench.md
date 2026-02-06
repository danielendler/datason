# Run Benchmarks

Run performance benchmarks and report results. Follow this workflow:

1. Run `just bench` or:
   ```
   uv run pytest tests/benchmarks/ --benchmark-only --benchmark-sort=mean
   ```

2. If benchmarks exist and pass, report:
   - Function name, mean time, std deviation
   - Compare against targets from CLAUDE.md:
     - Simple data: < 1ms
     - Complex data: < 5ms
     - Throughput: > 250K items/sec
   - Flag any regressions

3. If no benchmarks exist yet, create `tests/benchmarks/test_bench_core.py` with:
   - `bench_simple_dict` - 10 key flat dict
   - `bench_nested_dict` - 3 levels deep, 100 keys
   - `bench_list_of_dicts` - 1000 items
   - `bench_round_trip` - serialize then deserialize
   - `bench_with_datetimes` - 500 objects with datetime fields
   - `bench_with_uuids` - 500 objects with UUID fields

Always benchmark BEFORE and AFTER optimization attempts. Never optimize without data.
