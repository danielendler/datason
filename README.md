# datason

> **v2 rewrite in progress** — Plugin-based architecture, Python 3.10+, zero dependencies.

Zero-dependency Python serialization with intelligent type handling. Drop-in `json` replacement that handles datetime, UUID, numpy, pandas, PyTorch, TensorFlow, scikit-learn, and more.

## Install

```bash
pip install datason
```

## Usage

```python
import datason

# Serialize anything to JSON
data = {"timestamp": datetime.now(), "id": uuid4(), "scores": np.array([1, 2, 3])}
json_str = datason.dumps(data)

# Deserialize back to Python objects
original = datason.loads(json_str)
```

## API

```python
datason.dumps(obj)          # -> JSON string
datason.loads(s)            # -> Python object
datason.dump(obj, fp)       # -> write to file
datason.load(fp)            # -> read from file
```

## Status

This branch (`v2`) is a ground-up rewrite. See [CLAUDE.md](CLAUDE.md) for architecture and [LEARNINGS_AND_STRATEGY.md](LEARNINGS_AND_STRATEGY.md) for the v1 post-mortem.

## License

MIT
