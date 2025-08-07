#!/usr/bin/env python3
"""Streaming data loading example for datason.

This script demonstrates how to use the `stream_load()` function to efficiently
process large files without loading them entirely into memory.
"""

import gzip
import json
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List

import datason as ds


def generate_large_data(num_items: int = 1000) -> List[Dict[str, Any]]:
    """Generate a list of sample data items.

    Args:
        num_items: Number of items to generate

    Returns:
        List of dictionaries with sample data
    """
    return [
        {
            "id": i,
            "name": f"Item {i}",
            "value": i * 1.5,
            "active": i % 2 == 0,
            "tags": [f"tag{j}" for j in range(i % 5)],
        }
        for i in range(num_items)
    ]


def create_test_files() -> Dict[str, str]:
    """Create test files in different formats for demonstration.

    Returns:
        Dictionary with paths to the created files
    """
    data = generate_large_data(1000)
    temp_files = {}

    # Create JSONL file
    jsonl_path = os.path.join(tempfile.gettempdir(), f"temp_{os.urandom(8).hex()}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    temp_files["jsonl"] = jsonl_path

    # Create JSON array file
    json_path = os.path.join(tempfile.gettempdir(), f"temp_{os.urandom(8).hex()}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    temp_files["json"] = json_path

    # Create gzipped JSONL file
    gzipped_path = os.path.join(tempfile.gettempdir(), f"temp_{os.urandom(8).hex()}.jsonl.gz")
    with gzip.open(gzipped_path, "wt", encoding="utf-8") as gz:
        for item in data:
            gz.write(json.dumps(item) + "\n")
    temp_files["gzipped"] = gzipped_path

    return temp_files


def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single item from the stream.

    Args:
        item: The input item to process

    Returns:
        Processed item with additional fields
    """
    # Add a processed timestamp
    item["processed_at"] = datetime.utcnow().isoformat()
    # Convert value to string for demonstration
    item["value_str"] = f"${item['value']:.2f}"
    return item


def example_basic_streaming() -> None:
    """Demonstrate basic streaming with stream_load()."""
    print("=== Basic Streaming Example ===")

    # Create test files
    files = create_test_files()

    try:
        # Example 1: Basic streaming from JSONL
        print("\n1. Basic streaming from JSONL file:")
        with ds.stream_load(files["jsonl"]) as stream:
            # Process items one by one
            for i, item in enumerate(stream, 1):
                if i <= 3:  # Print first 3 items as example
                    print(f"  Item {i}: {item['id']} - {item['name']}")
                if i % 100 == 0:
                    print(f"  Processed {i} items...")
            print(f"  Total items processed: {stream.items_yielded}")

        # Example 2: Streaming with chunk processing
        print("\n2. Streaming with chunk processing:")
        with ds.stream_load(files["json"], format="json", chunk_processor=process_item) as stream:
            # Get first item to show processing
            first_item = next(iter(stream))
            print(f"  First item after processing: {first_item}")

        # Example 3: Streaming from gzipped file
        print("\n3. Streaming from gzipped JSONL file:")
        with ds.stream_load(files["gzipped"]) as stream:
            # Just count the items
            count = sum(1 for _ in stream)
            print(f"  Processed {count} items from gzipped file")

    finally:
        # Clean up temporary files
        for file_path in files.values():
            try:
                os.unlink(file_path)
            except OSError:
                pass


def main() -> None:
    """Run the streaming example."""
    print("=== DataSon Streaming Example ===\n")
    print("This example demonstrates how to use stream_load() for memory-efficient")
    print("processing of large files. It will create temporary files in different")
    print("formats and show how to process them using the streaming API.\n")

    example_basic_streaming()

    print("\n=== Example Complete ===")
    print("Temporary files have been cleaned up.")


if __name__ == "__main__":
    main()
