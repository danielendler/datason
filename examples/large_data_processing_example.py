"""Large Data Processing Example for datason.

This example demonstrates how to efficiently process large datasets with:
- Time-based filtering (e.g., last 5 years of data)
- Intelligent batching strategies
- Different processing modes for various algorithm needs
- Performance monitoring and optimization

Perfect for financial transaction data, time series analysis, and other large datasets.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
import tempfile

# Import the large data processor
from datason.large_data_processor import (
    LargeDataProcessor,
    create_financial_processor_config,
    create_time_series_processor_config,
    create_streaming_processor_config,
    ProcessingStrategy,
    TimeFilterConfig,
    BatchConfig,
    LargeDataConfig,
    MemoryManagement
)


def create_sample_transaction_data(num_records: int = 15000) -> List[Dict[str, Any]]:
    """Create sample financial transaction data for testing."""
    import random
    import uuid
    
    # Sample counterparties
    counterparties = [
        "Goldman Sachs", "JPMorgan Chase", "Bank of America", "Citigroup",
        "Wells Fargo", "Morgan Stanley", "Deutsche Bank", "HSBC",
        "Credit Suisse", "UBS", "Barclays", "BNP Paribas"
    ]
    
    # Generate data spanning 10 years (2014-2023)
    start_date = datetime(2014, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2023, 12, 31, tzinfo=timezone.utc)
    date_range = (end_date - start_date).days
    
    transactions = []
    
    for i in range(num_records):
        # Random date within the range
        random_days = random.randint(0, date_range)
        transaction_date = start_date + timedelta(days=random_days)
        
        transaction = {
            "transaction_id": str(uuid.uuid4()),
            "transaction_date": transaction_date.isoformat(),
            "counterparty": random.choice(counterparties),
            "amount": round(random.uniform(1000, 1000000), 2),
            "currency": random.choice(["USD", "EUR", "GBP", "JPY"]),
            "transaction_type": random.choice(["trade", "settlement", "payment", "transfer"]),
            "status": random.choice(["completed", "pending", "failed"]),
            "method_votes": [random.choice(["bank_transfer", "wire", "ach", "swift"])],
            "risk_score": round(random.uniform(0.1, 1.0), 3),
            "created_at": transaction_date.isoformat(),
            "processed_at": (transaction_date + timedelta(hours=random.randint(1, 24))).isoformat(),
        }
        transactions.append(transaction)
    
    # Sort by date to simulate realistic data
    transactions.sort(key=lambda x: x["transaction_date"])
    
    return transactions


def example_detection_algorithm(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Example detection algorithm that processes a batch of transactions.
    
    This simulates a financial detection algorithm that might look for:
    - High-risk transactions
    - Unusual patterns
    - Counterparty analysis
    """
    # Simulate some processing time
    time.sleep(0.01)  # 10ms processing time per batch
    
    high_risk_count = sum(1 for tx in batch if tx.get("risk_score", 0) > 0.8)
    large_amounts = sum(1 for tx in batch if tx.get("amount", 0) > 500000)
    
    counterparties = set(tx.get("counterparty", "unknown") for tx in batch)
    
    return {
        "batch_size": len(batch),
        "high_risk_transactions": high_risk_count,
        "large_amount_transactions": large_amounts,
        "unique_counterparties": len(counterparties),
        "total_amount": sum(tx.get("amount", 0) for tx in batch),
        "date_range": {
            "start": min(tx.get("transaction_date", "") for tx in batch),
            "end": max(tx.get("transaction_date", "") for tx in batch)
        }
    }


def example_ungrouped_algorithm(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Example algorithm that needs data ungrouped for time series analysis."""
    time.sleep(0.005)  # 5ms processing time
    
    # Time series analysis needs sequential data
    if len(batch) < 2:
        return {"trend": "insufficient_data", "batch_size": len(batch)}
    
    amounts = [tx.get("amount", 0) for tx in batch]
    
    # Simple trend analysis
    first_half = amounts[:len(amounts)//2]
    second_half = amounts[len(amounts)//2:]
    
    first_avg = sum(first_half) / len(first_half) if first_half else 0
    second_avg = sum(second_half) / len(second_half) if second_half else 0
    
    trend = "increasing" if second_avg > first_avg else "decreasing"
    
    return {
        "trend": trend,
        "batch_size": len(batch),
        "average_amount": sum(amounts) / len(amounts),
        "volatility": max(amounts) - min(amounts) if amounts else 0
    }


def demo_basic_usage() -> None:
    """Demonstrate basic usage of the large data processor."""
    print("🔍 Demo: Basic Large Data Processing")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample transaction data...")
    data = create_sample_transaction_data(5000)  # Smaller dataset for demo
    
    # Create a basic configuration
    config = LargeDataConfig(
        time_filter=TimeFilterConfig(
            years_back=5,  # Only process last 5 years
            timestamp_field="transaction_date"
        ),
        batch_config=BatchConfig(
            batch_size=500,
            strategy=ProcessingStrategy.SEQUENTIAL_UNGROUPED
        ),
        enable_progress=True
    )
    
    # Create processor
    processor = LargeDataProcessor(config)
    
    # Process the data
    print(f"\n🚀 Processing {len(data)} transactions...")
    results = []
    
    for result in processor.process_data(data, example_detection_algorithm):
        results.append(result)
    
    # Analyze results
    total_processed = sum(r["batch_size"] for r in results)
    total_high_risk = sum(r["high_risk_transactions"] for r in results)
    
    print(f"\n📈 Results Summary:")
    print(f"  • Processed {total_processed} transactions in {len(results)} batches")
    print(f"  • Found {total_high_risk} high-risk transactions")
    print(f"  • Processing rate: {processor.stats.records_per_second:.1f} records/sec")


def demo_financial_processing() -> None:
    """Demonstrate financial transaction processing with counterparty grouping."""
    print("\n💰 Demo: Financial Transaction Processing")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating financial transaction data...")
    data = create_sample_transaction_data(10000)
    
    # Use the financial processing configuration
    config = create_financial_processor_config(years_back=3)  # Last 3 years only
    
    processor = LargeDataProcessor(config)
    
    print(f"\n🏦 Processing {len(data)} transactions with counterparty grouping...")
    results = []
    
    for result in processor.process_data(data, example_detection_algorithm):
        results.append(result)
    
    # Analyze by counterparty
    counterparty_stats = {}
    for result in results:
        for counterparty in ["Goldman Sachs", "JPMorgan Chase", "Bank of America"]:
            if counterparty not in counterparty_stats:
                counterparty_stats[counterparty] = {
                    "batches": 0,
                    "transactions": 0,
                    "high_risk": 0
                }
    
    print(f"\n📊 Financial Processing Results:")
    print(f"  • Total batches: {len(results)}")
    print(f"  • Peak memory: {processor.stats.memory_peak_mb:.1f}MB")
    print(f"  • Average batch time: {processor.stats.average_processing_time_ms:.1f}ms")


def demo_time_series_processing() -> None:
    """Demonstrate time series processing with temporal chunking."""
    print("\n📈 Demo: Time Series Processing")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating time series data...")
    data = create_sample_transaction_data(8000)
    
    # Use time series configuration
    config = create_time_series_processor_config(years_back=7)
    
    processor = LargeDataProcessor(config)
    
    print(f"\n📅 Processing {len(data)} records with temporal chunking...")
    results = []
    
    for result in processor.process_data(data, example_ungrouped_algorithm):
        results.append(result)
    
    # Analyze trends
    increasing_periods = sum(1 for r in results if r.get("trend") == "increasing")
    decreasing_periods = sum(1 for r in results if r.get("trend") == "decreasing")
    
    print(f"\n📊 Time Series Results:")
    print(f"  • Temporal chunks processed: {len(results)}")
    print(f"  • Increasing trend periods: {increasing_periods}")
    print(f"  • Decreasing trend periods: {decreasing_periods}")
    print(f"  • Processing efficiency: {processor.stats.records_per_second:.1f} records/sec")


def demo_file_processing() -> None:
    """Demonstrate processing large files from disk."""
    print("\n💾 Demo: Large File Processing")
    print("=" * 50)
    
    # Create a large sample file
    print("📊 Creating large sample file...")
    data = create_sample_transaction_data(15000)  # 15K records as mentioned by user
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        temp_file = Path(f.name)
    
    try:
        # Get file size
        file_size_mb = temp_file.stat().st_size / (1024 * 1024)
        print(f"📁 Created file: {file_size_mb:.1f}MB with {len(data)} records")
        
        # Create configuration for large file processing
        config = LargeDataConfig(
            time_filter=TimeFilterConfig(
                years_back=5,
                timestamp_field="transaction_date"
            ),
            batch_config=BatchConfig(
                batch_size=1000,
                strategy=ProcessingStrategy.SEQUENTIAL_UNGROUPED,
                memory_management=MemoryManagement.BALANCED
            ),
            max_file_size_mb=5,  # Force chunked processing for demo
            enable_progress=True,
            enable_monitoring=True
        )
        
        processor = LargeDataProcessor(config)
        
        print(f"\n⚡ Processing large file...")
        results = []
        
        for result in processor.process_file(temp_file, example_detection_algorithm):
            results.append(result)
        
        print(f"\n📊 Large File Processing Results:")
        print(f"  • File size: {file_size_mb:.1f}MB")
        print(f"  • Results generated: {len(results)}")
        print(f"  • Peak memory usage: {processor.stats.memory_peak_mb:.1f}MB")
        print(f"  • Total processing time: {processor.stats.total_time_seconds:.2f}s")
        
        if processor.stats.warnings:
            print(f"  • Warnings: {len(processor.stats.warnings)}")
    
    finally:
        # Clean up
        temp_file.unlink()


def demo_custom_configuration() -> None:
    """Demonstrate creating custom configurations for specific needs."""
    print("\n⚙️ Demo: Custom Configuration")
    print("=" * 50)
    
    # Create a custom configuration for a specific use case
    custom_config = LargeDataConfig(
        time_filter=TimeFilterConfig(
            years_back=2,  # Only last 2 years
            timestamp_field="processed_at",  # Use processed_at instead of transaction_date
            fallback_timestamp_fields=["transaction_date", "created_at"],
            sort_by_timestamp=True,
            keep_records_without_timestamp=False  # Strict filtering
        ),
        batch_config=BatchConfig(
            batch_size=250,  # Smaller batches for precision
            dynamic_sizing=True,  # Adjust based on memory
            strategy=ProcessingStrategy.GROUPED_BY_COUNTERPARTY,
            memory_management=MemoryManagement.CONSERVATIVE
        ),
        max_file_size_mb=25,  # Conservative file size threshold
        enable_progress=True,
        enable_monitoring=True,
        slow_processing_threshold_ms=200,  # Very strict performance monitoring
        memory_warning_threshold_mb=256,   # Conservative memory usage
    )
    
    # Create sample data
    data = create_sample_transaction_data(3000)
    
    processor = LargeDataProcessor(custom_config)
    
    print(f"⚙️ Processing with custom configuration...")
    results = list(processor.process_data(data, example_detection_algorithm))
    
    print(f"\n📊 Custom Configuration Results:")
    print(f"  • Batches processed: {len(results)}")
    print(f"  • Records processed: {processor.stats.processed_records}")
    print(f"  • Records filtered by time: {processor.stats.filtered_records}")
    print(f"  • Processing rate: {processor.stats.records_per_second:.1f} records/sec")


def demo_performance_comparison() -> None:
    """Compare performance between different processing strategies."""
    print("\n🏃 Demo: Performance Comparison")
    print("=" * 50)
    
    # Create test data
    data = create_sample_transaction_data(5000)
    
    strategies = [
        ("Sequential Ungrouped", ProcessingStrategy.SEQUENTIAL_UNGROUPED),
        ("Grouped by Counterparty", ProcessingStrategy.GROUPED_BY_COUNTERPARTY),
        ("Temporal Chunks", ProcessingStrategy.TEMPORAL_CHUNKS),
    ]
    
    print("🔍 Testing different processing strategies...")
    
    for strategy_name, strategy in strategies:
        config = LargeDataConfig(
            time_filter=TimeFilterConfig(years_back=5),
            batch_config=BatchConfig(
                batch_size=500,
                strategy=strategy
            ),
            enable_progress=False,  # Disable progress for cleaner output
            enable_monitoring=True
        )
        
        processor = LargeDataProcessor(config)
        
        start_time = time.time()
        results = list(processor.process_data(data, example_detection_algorithm))
        end_time = time.time()
        
        print(f"\n📊 {strategy_name}:")
        print(f"  • Processing time: {end_time - start_time:.2f}s")
        print(f"  • Records/sec: {processor.stats.records_per_second:.1f}")
        print(f"  • Batches: {len(results)}")
        print(f"  • Peak memory: {processor.stats.memory_peak_mb:.1f}MB")


def main() -> None:
    """Run all the demos."""
    print("🚀 Large Data Processing Demos for datason")
    print("=" * 60)
    print("This demonstrates how to efficiently process large datasets")
    print("with time-based filtering and intelligent batching strategies.")
    print("Perfect for 15K+ record files as mentioned!")
    
    try:
        demo_basic_usage()
        demo_financial_processing()
        demo_time_series_processing()
        demo_file_processing()
        demo_custom_configuration()
        demo_performance_comparison()
        
        print("\n✅ All demos completed successfully!")
        print("\n💡 Key Benefits:")
        print("  • Time-based filtering reduces processing load")
        print("  • Intelligent batching optimizes memory usage")
        print("  • Different strategies for different algorithm needs")
        print("  • Built-in performance monitoring and warnings")
        print("  • Automatic memory management and cleanup")
        
        print("\n🎯 For your 15K+ file use case:")
        print("  • Use create_financial_processor_config() for counterparty grouping")
        print("  • Set years_back=5 to only process last 5 years")
        print("  • Enable progress monitoring to track performance")
        print("  • Consider TEMPORAL_CHUNKS for time series algorithms")
        
    except Exception as e:
        print(f"\n❌ Error running demos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()