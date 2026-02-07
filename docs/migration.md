# Migration from v1

datason v2 is a ground-up rewrite. The API has changed significantly.

## Quick Summary

| v1 | v2 |
|----|-----|
| `datason.serialize(obj)` | `datason.dumps(obj)` |
| `datason.deserialize(data)` | `datason.loads(json_str)` |
| `datason.dump_ml(obj)` | `datason.dumps(obj, **ml_config().__dict__)` |
| `datason.load_smart(data)` | `datason.loads(json_str)` |
| 100+ public functions | 5 functions: `dumps`, `loads`, `dump`, `load`, `config` |
| Python 3.8+ | Python 3.10+ |
| Monolithic core | Plugin-based architecture |

## Key Changes

1. **API reduced to 5 functions.** All the `dump_ml`, `load_smart`, `load_perfect`, `save_string` variants are gone. Use `datason.dumps()` with config overrides or presets instead.

2. **Configuration via kwargs or context manager**, not separate functions:
   ```python
   # v1
   datason.dump_ml(model_output)

   # v2
   from datason import ml_config
   with datason.config(**ml_config().__dict__):
       datason.dumps(model_output)
   # or inline:
   datason.dumps(model_output, date_format=DateFormat.UNIX_MS, fallback_to_string=True)
   ```

3. **Type metadata key changed** from various formats to a consistent `{"__datason_type__": "...", "__datason_value__": ...}` envelope.

4. **Python 3.10 minimum.** Drops 3.8/3.9 support. Uses `match/case` and `X | Y` union syntax.

5. **Thread-safe by default.** Config uses `contextvars.ContextVar`, registry uses `threading.Lock`.
