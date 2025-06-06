## Core Functions

The main serialization and deserialization functions that form the core of datason.

### serialize()

::: datason.serialize
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### deserialize()

::: datason.deserialize
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### auto_deserialize()

::: datason.auto_deserialize
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### safe_deserialize()

::: datason.safe_deserialize
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

## Chunked & Streaming Processing

Functions for handling large datasets efficiently.

### serialize_chunked()

::: datason.serialize_chunked
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### ChunkedSerializationResult

::: datason.ChunkedSerializationResult
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true

### StreamingSerializer

::: datason.StreamingSerializer
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true
      members:
        - serialize_async
        - stream_serialize

### estimate_memory_usage()

::: datason.estimate_memory_usage
    options:
      show_source: true
      show_signature: true
      show_signature_annotations: true
