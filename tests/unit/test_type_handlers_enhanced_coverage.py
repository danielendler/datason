"""Enhanced coverage tests for datason/type_handlers.py.

Targets missing lines to improve coverage from 68% to 90%+.
Missing lines: 16-17, 21-22, 48-49, 63, 77, 89, 116, 127-131, 143-147, 158, 170-175, 179, 205-213, 242-244, 271, 273, 275, 279, 301->308, 303, 305, 312-314, 329, 343, 372-373, 379-382
"""

import decimal
import enum
import uuid
from collections import namedtuple

import pytest

# Import without optional dependencies first
import datason.type_handlers as th
from datason.config import NanHandling, SerializationConfig, TypeCoercion

# Try importing optional dependencies
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


class TestTypeHandlerErrorPaths:
    """Test error paths and edge cases in TypeHandler."""

    def test_handle_decimal_exceptions(self):
        """Test decimal handling with overflow/value errors - line 48-49."""
        config = SerializationConfig()
        handler = th.TypeHandler(config)

        # Test overflow error with huge decimal
        huge_decimal = decimal.Decimal("1" + "0" * 400)
        result = handler.handle_decimal(huge_decimal)
        # Should return float(inf) or string
        assert isinstance(result, (float, str))

        # Test with NaN decimal
        nan_decimal = decimal.Decimal("NaN")
        result = handler.handle_decimal(nan_decimal)
        # Should return float(nan) or string
        assert isinstance(result, (float, str))

    def test_handle_complex_aggressive_coercion(self):
        """Test complex handling with aggressive coercion - line 63."""
        config = SerializationConfig(type_coercion=TypeCoercion.AGGRESSIVE)
        handler = th.TypeHandler(config)

        complex_num = complex(3.5, -2.1)
        result = handler.handle_complex(complex_num)
        assert result == [3.5, -2.1]

    def test_handle_uuid_conversion(self):
        """Test UUID handling - line 77."""
        config = SerializationConfig()
        handler = th.TypeHandler(config)

        test_uuid = uuid.uuid4()
        result = handler.handle_uuid(test_uuid)
        assert result == str(test_uuid)

    def test_handle_enum_values(self):
        """Test enum handling - line 89."""
        config = SerializationConfig()
        handler = th.TypeHandler(config)

        class TestEnum(enum.Enum):
            VALUE = "test_value"

        result = handler.handle_enum(TestEnum.VALUE)
        assert result == "test_value"

    def test_handle_namedtuple_error(self):
        """Test namedtuple error handling - line 116."""
        config = SerializationConfig()
        handler = th.TypeHandler(config)

        regular_tuple = (1, 2, 3)
        with pytest.raises(ValueError, match="Not a namedtuple"):
            handler.handle_namedtuple(regular_tuple)

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
    def test_handle_pandas_categorical_import_error(self):
        """Test pandas categorical without pandas - line 127-131."""
        config = SerializationConfig()
        handler = th.TypeHandler(config)

        # Temporarily disable pandas
        original_pd = th.pd
        th.pd = None

        try:
            with pytest.raises(ImportError, match="pandas not available"):
                handler.handle_pandas_categorical([1, 2, 3])
        finally:
            th.pd = original_pd

    def test_handle_set_unsortable(self):
        """Test set handling with unsortable items - line 143-147."""
        config = SerializationConfig()
        handler = th.TypeHandler(config)

        # Mix of sortable and unsortable types
        mixed_set = {1, "string", complex(1, 2)}
        result = handler.handle_set(mixed_set)
        assert isinstance(result, list)
        assert len(result) == 3
        assert set(result) == mixed_set

    def test_handle_set_sortable(self):
        """Test set handling with sortable items - line 158."""
        config = SerializationConfig()
        handler = th.TypeHandler(config)

        sortable_set = {3, 1, 4, 5}
        result = handler.handle_set(sortable_set)
        assert isinstance(result, list)
        assert result == sorted(sortable_set)

    def test_handle_bytes_decode_error(self):
        """Test bytes handling with decode error - line 170-175."""
        config = SerializationConfig()
        handler = th.TypeHandler(config)

        # Invalid UTF-8 bytes
        invalid_bytes = b"\xff\xfe\xfd"
        result = handler.handle_bytes(invalid_bytes)
        assert result == invalid_bytes.hex()

    def test_handle_bytearray(self):
        """Test bytearray handling - line 179."""
        config = SerializationConfig()
        handler = th.TypeHandler(config)

        test_bytearray = bytearray(b"hello")
        result = handler.handle_bytearray(test_bytearray)
        assert result == "hello"

    def test_handle_nan_value_strategies(self):
        """Test NaN handling strategies - line 205-213."""
        # Test NULL strategy
        config_null = SerializationConfig(nan_handling=NanHandling.NULL)
        handler_null = th.TypeHandler(config_null)
        assert handler_null.handle_nan_value(float("nan")) is None

        # Test STRING strategy with __name__
        config_string = SerializationConfig(nan_handling=NanHandling.STRING)
        handler_string = th.TypeHandler(config_string)

        class NamedObject:
            __name__ = "TestObject"

        named_obj = NamedObject()
        result = handler_string.handle_nan_value(named_obj)
        assert result == "<TestObject>"

        # Test STRING strategy without __name__
        result = handler_string.handle_nan_value(42)
        assert result == "42"

        # Test KEEP strategy
        config_keep = SerializationConfig(nan_handling=NanHandling.KEEP)
        handler_keep = th.TypeHandler(config_keep)
        nan_val = float("nan")
        result = handler_keep.handle_nan_value(nan_val)
        assert result is nan_val

        # Test DROP strategy
        config_drop = SerializationConfig(nan_handling=NanHandling.DROP)
        handler_drop = th.TypeHandler(config_drop)
        result = handler_drop.handle_nan_value(float("nan"))
        assert result is None

    def test_is_namedtuple_edge_cases(self):
        """Test namedtuple detection - line 242-244."""
        config = SerializationConfig()
        handler = th.TypeHandler(config)

        # Proper namedtuple
        Point = namedtuple("Point", ["x", "y"])
        point = Point(1, 2)
        assert handler.is_namedtuple(point) is True

        # Regular tuple
        regular_tuple = (1, 2)
        assert handler.is_namedtuple(regular_tuple) is False

        # Fake namedtuple (not a tuple)
        class FakeNamedTuple:
            _fields = ("x", "y")
            _field_defaults = {}

            def _asdict(self):
                return {}

        fake = FakeNamedTuple()
        assert handler.is_namedtuple(fake) is False


class TestTypeHandlerCustomization:
    """Test custom serializers and type metadata."""

    def test_custom_serializers_priority(self):
        """Test custom serializers take precedence - line 271, 273, 275."""

        def custom_handler(obj):
            return f"custom_{obj}"

        config = SerializationConfig(custom_serializers={int: custom_handler})
        handler = th.TypeHandler(config)

        # Should return custom handler
        result_handler = handler.get_type_handler(42)
        assert result_handler is custom_handler

        # Should return None for non-custom types
        str_handler = handler.get_type_handler("test")
        assert str_handler is None

    def test_type_metadata_skipping(self):
        """Test type metadata skipping - line 279, 301->308, 303, 305."""
        # Mock config with type hints
        config = SerializationConfig()
        config.include_type_hints = True
        handler = th.TypeHandler(config)

        # These should return None with type hints enabled
        assert handler.get_type_handler(decimal.Decimal("1.23")) is None
        assert handler.get_type_handler(complex(1, 2)) is None
        assert handler.get_type_handler(uuid.uuid4()) is None
        assert handler.get_type_handler({1, 2, 3}) is None
        assert handler.get_type_handler(range(5)) is None

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
    def test_pandas_categorical_handler(self):
        """Test pandas categorical handler detection - line 312-314."""
        config = SerializationConfig()
        handler = th.TypeHandler(config)

        categorical = pd.Categorical(["a", "b", "c"])
        cat_handler = handler.get_type_handler(categorical)
        # Check that the handler function is returned (avoid 'is' comparison)
        assert cat_handler == handler.handle_pandas_categorical
        assert callable(cat_handler)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_is_nan_like_none(self):
        """Test is_nan_like with None - line 329."""
        assert th.is_nan_like(None) is True

    def test_is_nan_like_float_nan(self):
        """Test is_nan_like with float NaN."""
        assert th.is_nan_like(float("nan")) is True
        assert th.is_nan_like(1.5) is False

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy not available")
    def test_is_nan_like_numpy(self):
        """Test is_nan_like with numpy types - line 343."""
        assert th.is_nan_like(np.float64("nan")) is True
        assert th.is_nan_like(np.datetime64("NaT")) is True
        assert th.is_nan_like(np.float64(1.5)) is False

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
    def test_is_nan_like_pandas_exception(self):
        """Test pandas exception handling - line 372-373."""

        class ProblematicObject:
            def __len__(self):
                raise ValueError("Test error")

        problematic = ProblematicObject()
        # Should not raise and return False
        assert th.is_nan_like(problematic) is False

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy not available")
    def test_normalize_numpy_types(self):
        """Test numpy type normalization - line 379-382."""
        # Test bool
        assert th.normalize_numpy_types(np.bool_(True)) is True

        # Test integer
        assert th.normalize_numpy_types(np.int64(42)) == 42

        # Test float with NaN/inf
        assert th.normalize_numpy_types(np.float64("nan")) is None
        assert th.normalize_numpy_types(np.float64("inf")) is None
        assert th.normalize_numpy_types(np.float64(3.14)) == 3.14

        # Test string
        assert th.normalize_numpy_types(np.str_("hello")) == "hello"

        # Test bytes
        assert th.normalize_numpy_types(np.bytes_(b"hello")) == b"hello"

        # Test passthrough
        regular_obj = [1, 2, 3]
        assert th.normalize_numpy_types(regular_obj) is regular_obj


class TestGetObjectInfo:
    """Test get_object_info function."""

    def test_get_object_info_basic(self):
        """Test get_object_info basic functionality."""
        info = th.get_object_info(42)
        assert info["type"] == "int"
        assert info["module"] == "builtins"
        assert "int" in info["mro"]
        assert info["is_callable"] is False
        assert info["has_dict"] is False

    def test_get_object_info_collections(self):
        """Test get_object_info with collections."""
        # Test list
        test_list = [1, "str", 3.14]
        info = th.get_object_info(test_list)
        assert info["type"] == "list"
        assert info["size"] == 3
        assert "sample_types" in info

        # Test dict
        test_dict = {"key": "value", 42: [1, 2]}
        info = th.get_object_info(test_dict)
        assert info["type"] == "dict"
        assert info["size"] == 2
        assert "sample_key_types" in info
        assert "sample_value_types" in info

    def test_get_object_info_exception(self):
        """Test get_object_info exception handling."""

        class BadLength:
            def __len__(self):
                raise RuntimeError("Bad length")

        bad_obj = BadLength()
        info = th.get_object_info(bad_obj)
        assert info["size"] is None


class TestImportFallbacks:
    """Test import fallback scenarios."""

    def test_pandas_fallback(self):
        """Test pandas import fallback - line 16-17."""
        original_pd = th.pd
        th.pd = None

        try:
            # Should not crash
            assert th.pd is None
            config = SerializationConfig()
            handler = th.TypeHandler(config)
            result = handler.get_type_handler("test")
            assert result is None
        finally:
            th.pd = original_pd

    def test_numpy_fallback(self):
        """Test numpy import fallback - line 21-22."""
        original_np = th.np
        th.np = None

        try:
            # Should not crash
            assert th.np is None
            assert th.is_nan_like(None) is True
            assert th.normalize_numpy_types(42) == 42
        finally:
            th.np = original_np

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not available")
    def test_pandas_isna_strings_bytes(self):
        """Test pandas isna with strings and bytes."""
        # These should not be considered NaN-like
        assert th.is_nan_like("test_string") is False
        assert th.is_nan_like(b"test_bytes") is False

        # Test with pandas NA if available
        if hasattr(pd, "NA"):
            assert th.is_nan_like(pd.NA) is True
