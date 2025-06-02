"""Pickle Bridge - Convert legacy pickle files to portable datason JSON.

This module provides safe conversion of pickle files to JSON format,
addressing the ML community's need to migrate from pickle-based workflows
to portable, readable JSON serialization.

Features:
- Security-first approach with class whitelisting
- Zero new dependencies (uses stdlib pickle only)
- Leverages existing datason type handlers
- Supports bulk directory conversion
- Streaming for large pickle files
"""

import pickle  # nosec B403
import warnings
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Set, Union

from .config import SerializationConfig, get_ml_config
from .core import SecurityError, serialize


class PickleSecurityError(SecurityError):
    """Raised when unsafe pickle content is detected."""


class PickleBridge:
    """Main class for converting pickle files to datason JSON.

    Provides safe, configurable conversion of pickle files with security
    controls and performance optimizations.
    """

    # Default safe classes for ML workflows
    DEFAULT_SAFE_CLASSES: ClassVar[Set[str]] = {
        # NumPy
        "numpy.ndarray",
        "numpy.dtype",
        "numpy.matrix",
        # Pandas
        "pandas.core.frame.DataFrame",
        "pandas.core.series.Series",
        "pandas.core.index.Index",
        "pandas.core.indexes.base.Index",
        "pandas.core.indexes.range.RangeIndex",
        "pandas.core.indexes.numeric.Int64Index",
        "pandas.core.indexes.numeric.Float64Index",
        "pandas.core.indexes.datetimes.DatetimeIndex",
        "pandas.core.arrays.categorical.Categorical",
        # Scikit-learn
        "sklearn.linear_model._base.LinearRegression",
        "sklearn.linear_model._logistic.LogisticRegression",
        "sklearn.ensemble._forest.RandomForestClassifier",
        "sklearn.ensemble._forest.RandomForestRegressor",
        "sklearn.tree._classes.DecisionTreeClassifier",
        "sklearn.tree._classes.DecisionTreeRegressor",
        "sklearn.svm._classes.SVC",
        "sklearn.svm._classes.SVR",
        "sklearn.naive_bayes.GaussianNB",
        "sklearn.neighbors._classification.KNeighborsClassifier",
        "sklearn.cluster._kmeans.KMeans",
        "sklearn.decomposition._pca.PCA",
        "sklearn.preprocessing._encoders.OneHotEncoder",
        "sklearn.preprocessing._data.StandardScaler",
        "sklearn.preprocessing._data.MinMaxScaler",
        "sklearn.model_selection._split.train_test_split",
        "sklearn.pipeline.Pipeline",
        # PyTorch basic types
        "torch.Tensor",
        "torch.nn.modules.module.Module",
        # Basic Python types that are always safe
        "builtins.dict",
        "builtins.list",
        "builtins.tuple",
        "builtins.set",
        "builtins.frozenset",
        "builtins.str",
        "builtins.int",
        "builtins.float",
        "builtins.complex",
        "builtins.bool",
        "builtins.bytes",
        "builtins.NoneType",
        # Standard library types
        "datetime.datetime",
        "datetime.date",
        "datetime.time",
        "datetime.timedelta",
        "uuid.UUID",
        "decimal.Decimal",
        "fractions.Fraction",
        "collections.OrderedDict",
        "collections.defaultdict",
        "collections.Counter",
        "collections.deque",
    }

    def __init__(
        self,
        safe_classes: Optional[Set[str]] = None,
        config: Optional[SerializationConfig] = None,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB default
    ) -> None:
        """Initialize the PickleBridge.

        Args:
            safe_classes: Set of allowed class names. If None, uses DEFAULT_SAFE_CLASSES.
            config: Serialization config. If None, uses ML config.
            max_file_size: Maximum pickle file size in bytes.
        """
        self.safe_classes = safe_classes or self.DEFAULT_SAFE_CLASSES.copy()
        self.config = config or get_ml_config()
        self.max_file_size = max_file_size
        self._conversion_stats = {
            "files_processed": 0,
            "files_successful": 0,
            "files_failed": 0,
            "total_size_bytes": 0,
        }

    def add_safe_class(self, class_name: str) -> None:
        """Add a class to the safe classes whitelist.

        Args:
            class_name: Full module.class name (e.g., 'sklearn.ensemble.RandomForestClassifier')
        """
        self.safe_classes.add(class_name)

    def add_safe_module(self, module_prefix: str) -> None:
        """Add all classes from a module to the safe classes whitelist.

        Args:
            module_prefix: Module prefix (e.g., 'sklearn', 'numpy')
        """
        # This is a convenience method - in practice, users should be specific
        # about which classes they trust for security reasons
        warnings.warn(
            f"Adding entire module '{module_prefix}' to safe classes. "
            "For better security, consider adding specific classes instead.",
            stacklevel=2,
        )
        # Store module prefix for checking during unpickling
        self.safe_classes.add(f"{module_prefix}.*")

    def _safe_unpickler(self, file_obj: Any) -> Any:
        """Create a secure unpickler with class restrictions."""
        safe_classes = self.safe_classes  # Capture in closure

        class SafeUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str) -> Any:
                full_name = f"{module}.{name}"

                # Check exact matches first
                if full_name in safe_classes:
                    return super().find_class(module, name)

                # Check module wildcards
                for safe_class in safe_classes:
                    if safe_class.endswith(".*") and full_name.startswith(safe_class[:-1]):
                        return super().find_class(module, name)

                # Class not in whitelist
                raise PickleSecurityError(
                    f"Attempted to unpickle unauthorized class: {full_name}. "
                    f"Add to safe_classes if this class is trusted."
                )

        return SafeUnpickler(file_obj)

    def from_pickle_file(self, pickle_path: Union[str, Path]) -> Dict[str, Any]:
        """Convert a pickle file to datason JSON format.

        Args:
            pickle_path: Path to the pickle file

        Returns:
            Dictionary containing the serialized data and metadata

        Raises:
            PickleSecurityError: If unsafe classes are detected
            FileNotFoundError: If pickle file doesn't exist
            SecurityError: If file is too large
        """
        pickle_path = Path(pickle_path)

        # Check if file exists first
        if not pickle_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

        # Security check: file size
        file_size = pickle_path.stat().st_size
        if file_size > self.max_file_size:
            raise SecurityError(
                f"Pickle file size ({file_size:,} bytes) exceeds maximum ({self.max_file_size:,} bytes)"
            )

        self._conversion_stats["files_processed"] += 1
        self._conversion_stats["total_size_bytes"] += file_size

        try:
            # Load pickle with security restrictions
            with pickle_path.open("rb") as f:
                unpickler = self._safe_unpickler(f)
                obj = unpickler.load()

            # Convert to datason JSON
            json_data = serialize(obj, config=self.config)

            # Create result with metadata
            result = {
                "data": json_data,
                "metadata": {
                    "source_file": str(pickle_path),
                    "source_size_bytes": file_size,
                    "conversion_timestamp": serialize(__import__("datetime").datetime.now(), config=self.config),
                    "datason_version": "0.3.0",  # Current version
                    "safe_classes_used": sorted(self.safe_classes),
                },
            }

            self._conversion_stats["files_successful"] += 1
            return result

        except (PickleSecurityError, SecurityError):
            self._conversion_stats["files_failed"] += 1
            raise
        except Exception as e:
            self._conversion_stats["files_failed"] += 1
            # Wrap other exceptions for clarity
            raise PickleSecurityError(f"Failed to convert pickle file {pickle_path}: {e}") from e

    def from_pickle_bytes(self, pickle_data: bytes) -> Dict[str, Any]:
        """Convert pickle bytes to datason JSON format.

        Args:
            pickle_data: Raw pickle bytes

        Returns:
            Dictionary containing the serialized data and metadata
        """
        if len(pickle_data) > self.max_file_size:
            raise SecurityError(
                f"Pickle data size ({len(pickle_data):,} bytes) exceeds maximum ({self.max_file_size:,} bytes)"
            )

        try:
            # Load pickle with security restrictions
            import io

            with io.BytesIO(pickle_data) as f:
                unpickler = self._safe_unpickler(f)
                obj = unpickler.load()

            # Convert to datason JSON
            json_data = serialize(obj, config=self.config)

            return {
                "data": json_data,
                "metadata": {
                    "source_size_bytes": len(pickle_data),
                    "conversion_timestamp": serialize(__import__("datetime").datetime.now(), config=self.config),
                    "datason_version": "0.3.0",
                },
            }

        except (PickleSecurityError, SecurityError):
            raise
        except Exception as e:
            raise PickleSecurityError(f"Failed to convert pickle data: {e}") from e

    def convert_directory(
        self,
        source_dir: Union[str, Path],
        target_dir: Union[str, Path],
        pattern: str = "*.pkl",
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Convert all pickle files in a directory to JSON format.

        Args:
            source_dir: Directory containing pickle files
            target_dir: Directory to save JSON files
            pattern: File pattern to match (default: "*.pkl")
            overwrite: Whether to overwrite existing JSON files

        Returns:
            Dictionary with conversion statistics
        """
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)

        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        # Create target directory if it doesn't exist
        target_dir.mkdir(parents=True, exist_ok=True)

        # Reset stats for this batch
        batch_stats = {
            "files_found": 0,
            "files_converted": 0,
            "files_skipped": 0,
            "files_failed": 0,
            "errors": [],
        }

        # Find all matching files
        pickle_files = list(source_dir.glob(pattern))
        batch_stats["files_found"] = len(pickle_files)

        for pickle_file in pickle_files:
            json_file = target_dir / f"{pickle_file.stem}.json"

            # Skip if file exists and overwrite is False
            if json_file.exists() and not overwrite:
                batch_stats["files_skipped"] += 1
                continue

            try:
                # Convert pickle to JSON
                result = self.from_pickle_file(pickle_file)

                # Save JSON file
                import json

                with json_file.open("w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                batch_stats["files_converted"] += 1

            except Exception as e:
                batch_stats["files_failed"] += 1
                batch_stats["errors"].append(
                    {
                        "file": str(pickle_file),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )

        return batch_stats

    def get_conversion_stats(self) -> Dict[str, Any]:
        """Get conversion statistics for this bridge instance."""
        return self._conversion_stats.copy()


# Convenience functions for quick access
def from_pickle(
    pickle_path: Union[str, Path],
    safe_classes: Optional[Set[str]] = None,
    config: Optional[SerializationConfig] = None,
) -> Dict[str, Any]:
    """Convert a pickle file to datason JSON (convenience function).

    Args:
        pickle_path: Path to pickle file
        safe_classes: Set of allowed class names
        config: Serialization configuration

    Returns:
        Dictionary with converted data and metadata
    """
    bridge = PickleBridge(safe_classes=safe_classes, config=config)
    return bridge.from_pickle_file(pickle_path)


def convert_pickle_directory(
    source_dir: Union[str, Path],
    target_dir: Union[str, Path],
    pattern: str = "*.pkl",
    safe_classes: Optional[Set[str]] = None,
    config: Optional[SerializationConfig] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Convert all pickle files in a directory (convenience function).

    Args:
        source_dir: Source directory with pickle files
        target_dir: Target directory for JSON files
        pattern: File pattern to match
        safe_classes: Set of allowed class names
        config: Serialization configuration
        overwrite: Whether to overwrite existing files

    Returns:
        Dictionary with conversion statistics
    """
    bridge = PickleBridge(safe_classes=safe_classes, config=config)
    return bridge.convert_directory(source_dir, target_dir, pattern, overwrite)


def get_ml_safe_classes() -> Set[str]:
    """Get the default set of ML-safe classes for pickle conversion.

    Returns:
        Set of class names considered safe for ML workflows
    """
    return PickleBridge.DEFAULT_SAFE_CLASSES.copy()
