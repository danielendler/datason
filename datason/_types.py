"""Type constants and categories for datason serialization."""

# Types that json.dumps handles natively — no plugin needed
JSON_BASIC_TYPES: tuple[type, ...] = (str, int, float, bool, type(None))

# Reserved metadata keys for type-annotated serialization
TYPE_METADATA_KEY: str = "__datason_type__"
VALUE_METADATA_KEY: str = "__datason_value__"

# Security defaults
DEFAULT_MAX_DEPTH: int = 50
DEFAULT_MAX_SIZE: int = 100_000
DEFAULT_MAX_STRING_LENGTH: int = 1_000_000
