import json
import gzip
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Union

from .core import serialize
from .deserializers import deserialize


def _detect_format(path: Path, fmt: Optional[str]) -> str:
    """Detect file format based on extension or provided parameter."""
    if fmt:
        return fmt
    suffixes = path.suffixes
    if suffixes and suffixes[-1] == ".gz" and len(suffixes) > 1:
        ext = suffixes[-2]
    elif suffixes:
        ext = suffixes[-1]
    else:
        ext = ""
    if ext in {".jsonl", ".ndjson"}:
        return "jsonl"
    return "json"


def _open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def save(obj: Iterable[Any], file_path: Union[str, Path], *, format: Optional[str] = None) -> None:
    """Save an object or iterable to a file in JSON or JSONL format."""
    path = Path(file_path)
    fmt = _detect_format(path, format)

    if fmt == "jsonl":
        if not (hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, dict))):
            iterable = [obj]
        else:
            iterable = obj  # type: ignore[assignment]
        with _open_text(path, "w") as f:
            for item in iterable:
                json.dump(serialize(item), f, ensure_ascii=False)
                f.write("\n")
    elif fmt == "json":
        with _open_text(path, "w") as f:
            json.dump(serialize(obj), f, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def load(file_path: Union[str, Path], *, format: Optional[str] = None) -> Iterator[Any]:
    """Load objects from a JSON or JSONL file."""
    path = Path(file_path)
    fmt = _detect_format(path, format)

    def _iter_jsonl() -> Iterator[Any]:
        with _open_text(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield deserialize(json.loads(line))

    if fmt == "jsonl":
        return _iter_jsonl()
    if fmt == "json":
        with _open_text(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return (deserialize(item) for item in data)
        return iter([deserialize(data)])
    raise ValueError(f"Unsupported format: {fmt}")
