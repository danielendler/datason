"""Large Data Processor for efficient handling of big datasets.

This module provides intelligent processing strategies for large files with:
- Time-based filtering (e.g., last N years of data)
- Intelligent batching for different algorithm types
- Memory-efficient streaming processing
- Performance optimization and monitoring
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Set, Union, Callable
import time
import os

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

from .config import SerializationConfig, get_performance_config
from .core import serialize_chunked, deserialize_chunked_file
from .utils import find_data_anomalies


class ProcessingStrategy(Enum):
    """Different processing strategies for various algorithm needs."""
    
    # For algorithms that need all data ungrouped (e.g., time series analysis)
    SEQUENTIAL_UNGROUPED = "sequential_ungrouped"
    
    # For algorithms that work better with counterparty grouping
    GROUPED_BY_COUNTERPARTY = "grouped_by_counterparty"
    
    # For algorithms that can work with temporal chunks
    TEMPORAL_CHUNKS = "temporal_chunks"
    
    # For streaming algorithms that process one record at a time
    STREAMING = "streaming"
    
    # For algorithms that need the full dataset in memory
    FULL_DATASET = "full_dataset"


class MemoryManagement(Enum):
    """Memory management strategies."""
    
    CONSERVATIVE = "conservative"  # Aggressive cleanup, smaller batches
    BALANCED = "balanced"         # Balance between speed and memory
    AGGRESSIVE = "aggressive"     # Large batches, minimal cleanup


@dataclass
class TimeFilterConfig:
    """Configuration for time-based filtering."""
    
    # How many years to look back from the latest transaction
    years_back: Optional[int] = None
    
    # Specific date range
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Field name that contains the timestamp
    timestamp_field: str = "timestamp"
    
    # Alternative timestamp fields to try if primary field not found
    fallback_timestamp_fields: List[str] = field(default_factory=lambda: [
        "date", "created_at", "transaction_date", "processed_at", "updated_at"
    ])
    
    # Whether to sort by timestamp after filtering
    sort_by_timestamp: bool = True
    
    # Whether to keep records with missing timestamps
    keep_records_without_timestamp: bool = False


@dataclass
class BatchConfig:
    """Configuration for batching strategies."""
    
    # Base batch size
    batch_size: int = 1000
    
    # Dynamic batch sizing based on available memory
    dynamic_sizing: bool = True
    
    # Maximum batch size (safety limit)
    max_batch_size: int = 10000
    
    # Minimum batch size (efficiency limit)
    min_batch_size: int = 100
    
    # Memory usage threshold for batch size adjustment (MB)
    memory_threshold_mb: int = 512
    
    # Processing strategy
    strategy: ProcessingStrategy = ProcessingStrategy.SEQUENTIAL_UNGROUPED
    
    # Memory management strategy
    memory_management: MemoryManagement = MemoryManagement.BALANCED


@dataclass
class LargeDataConfig:
    """Configuration for large data processing."""
    
    # Time filtering configuration
    time_filter: Optional[TimeFilterConfig] = None
    
    # Batching configuration
    batch_config: BatchConfig = field(default_factory=BatchConfig)
    
    # Serialization configuration (uses performance config by default)
    serialization_config: Optional[SerializationConfig] = None
    
    # Maximum file size in MB before forcing chunked processing
    max_file_size_mb: int = 100
    
    # Enable progress reporting
    enable_progress: bool = True
    
    # Enable performance monitoring
    enable_monitoring: bool = True
    
    # Warning thresholds
    slow_processing_threshold_ms: int = 1000  # Warn if processing takes >1s per batch
    memory_warning_threshold_mb: int = 1024   # Warn if memory usage >1GB
    
    # Optimization flags
    enable_type_caching: bool = True
    enable_string_interning: bool = True
    enable_early_termination: bool = True  # Stop processing if conditions met


@dataclass
class ProcessingStats:
    """Statistics for processing operations."""
    
    total_records: int = 0
    processed_records: int = 0
    filtered_records: int = 0
    skipped_records: int = 0
    
    batches_processed: int = 0
    
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    memory_peak_mb: float = 0.0
    memory_current_mb: float = 0.0
    
    processing_times_ms: List[float] = field(default_factory=list)
    
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def total_time_seconds(self) -> float:
        """Total processing time in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def average_processing_time_ms(self) -> float:
        """Average processing time per batch in milliseconds."""
        return sum(self.processing_times_ms) / len(self.processing_times_ms) if self.processing_times_ms else 0.0
    
    @property
    def records_per_second(self) -> float:
        """Processing rate in records per second."""
        if self.total_time_seconds > 0:
            return self.processed_records / self.total_time_seconds
        return 0.0


class LargeDataProcessor:
    """Processor for handling large datasets efficiently."""
    
    def __init__(self, config: Optional[LargeDataConfig] = None):
        """Initialize the processor with configuration."""
        self.config = config or LargeDataConfig()
        self.stats = ProcessingStats()
        
        # Set up serialization config if not provided
        if self.config.serialization_config is None:
            self.config.serialization_config = get_performance_config()
        
        # Type caching for performance
        self._type_cache: Dict[type, str] = {}
        self._string_cache: Dict[str, Any] = {}
        
        # Memory monitoring
        if psutil is not None:
            self._process = psutil.Process(os.getpid())
        else:
            self._process = None
    
    def process_file(
        self, 
        file_path: Union[str, Path],
        processor_func: Callable[[List[Dict[str, Any]]], Any],
        **kwargs: Any
    ) -> Generator[Any, None, None]:
        """Process a large file with intelligent strategies.
        
        Args:
            file_path: Path to the file to process
            processor_func: Function to process each batch of data
            **kwargs: Additional arguments for the processor function
            
        Yields:
            Results from the processor function for each batch
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file size and decide processing strategy
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        if file_size_mb > self.config.max_file_size_mb:
            if self.config.enable_progress:
                print(f"📊 Large file detected ({file_size_mb:.1f}MB), using chunked processing")
            
            yield from self._process_large_file_chunked(file_path, processor_func, **kwargs)
        else:
            if self.config.enable_progress:
                print(f"📊 Processing file ({file_size_mb:.1f}MB) in memory")
            
            yield from self._process_file_in_memory(file_path, processor_func, **kwargs)
    
    def process_data(
        self,
        data: List[Dict[str, Any]],
        processor_func: Callable[[List[Dict[str, Any]]], Any],
        **kwargs: Any
    ) -> Generator[Any, None, None]:
        """Process data with intelligent batching and filtering.
        
        Args:
            data: List of data records
            processor_func: Function to process each batch
            **kwargs: Additional arguments for the processor function
            
        Yields:
            Results from the processor function for each batch
        """
        self.stats = ProcessingStats()
        self.stats.start_time = datetime.now(timezone.utc)
        self.stats.total_records = len(data)
        
        try:
            # Apply time filtering if configured
            if self.config.time_filter:
                data = self._apply_time_filter(data)
                if self.config.enable_progress:
                    print(f"🕒 Time filter applied: {len(data)}/{self.stats.total_records} records remaining")
            
            # Apply processing strategy
            if self.config.batch_config.strategy == ProcessingStrategy.GROUPED_BY_COUNTERPARTY:
                yield from self._process_grouped_by_counterparty(data, processor_func, **kwargs)
            elif self.config.batch_config.strategy == ProcessingStrategy.TEMPORAL_CHUNKS:
                yield from self._process_temporal_chunks(data, processor_func, **kwargs)
            elif self.config.batch_config.strategy == ProcessingStrategy.STREAMING:
                yield from self._process_streaming(data, processor_func, **kwargs)
            else:  # SEQUENTIAL_UNGROUPED or FULL_DATASET
                yield from self._process_sequential(data, processor_func, **kwargs)
        
        finally:
            self.stats.end_time = datetime.now(timezone.utc)
            if self.config.enable_progress:
                self._print_final_stats()
    
    def _apply_time_filter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply time-based filtering to the data."""
        if not self.config.time_filter:
            return data
        
        filter_config = self.config.time_filter
        filtered_data = []
        
        # Find the latest timestamp if using years_back
        latest_timestamp = None
        if filter_config.years_back:
            latest_timestamp = self._find_latest_timestamp(data)
            if latest_timestamp:
                cutoff_date = latest_timestamp - timedelta(days=filter_config.years_back * 365)
        
        # Determine date range
        start_date = filter_config.start_date
        end_date = filter_config.end_date
        
        if filter_config.years_back and latest_timestamp:
            end_date = latest_timestamp
            start_date = cutoff_date
        
        for record in data:
            timestamp = self._extract_timestamp(record)
            
            if timestamp is None:
                if filter_config.keep_records_without_timestamp:
                    filtered_data.append(record)
                    self.stats.processed_records += 1
                else:
                    self.stats.skipped_records += 1
                continue
            
            # Check if timestamp is in range
            include_record = True
            
            if start_date and timestamp < start_date:
                include_record = False
            if end_date and timestamp > end_date:
                include_record = False
            
            if include_record:
                filtered_data.append(record)
                self.stats.processed_records += 1
            else:
                self.stats.filtered_records += 1
        
        # Sort by timestamp if requested
        if filter_config.sort_by_timestamp and filtered_data:
            filtered_data.sort(key=lambda x: self._extract_timestamp(x) or datetime.min.replace(tzinfo=timezone.utc))
        
        return filtered_data
    
    def _find_latest_timestamp(self, data: List[Dict[str, Any]]) -> Optional[datetime]:
        """Find the latest timestamp in the data."""
        latest = None
        
        for record in data:
            timestamp = self._extract_timestamp(record)
            if timestamp and (latest is None or timestamp > latest):
                latest = timestamp
        
        return latest
    
    def _extract_timestamp(self, record: Dict[str, Any]) -> Optional[datetime]:
        """Extract timestamp from a record."""
        filter_config = self.config.time_filter
        if not filter_config:
            return None
        
        # Try primary timestamp field
        timestamp_value = record.get(filter_config.timestamp_field)
        
        # Try fallback fields if primary not found
        if timestamp_value is None:
            for field in filter_config.fallback_timestamp_fields:
                timestamp_value = record.get(field)
                if timestamp_value is not None:
                    break
        
        if timestamp_value is None:
            return None
        
        # Convert to datetime if needed
        if isinstance(timestamp_value, str):
            try:
                # Try ISO format first
                if 'T' in timestamp_value:
                    return datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                else:
                    # Try common date formats
                    for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y']:
                        try:
                            dt = datetime.strptime(timestamp_value, fmt)
                            return dt.replace(tzinfo=timezone.utc)
                        except ValueError:
                            continue
            except ValueError:
                pass
        elif isinstance(timestamp_value, (int, float)):
            # Assume Unix timestamp
            try:
                if timestamp_value > 1e10:  # Milliseconds
                    return datetime.fromtimestamp(timestamp_value / 1000, tz=timezone.utc)
                else:  # Seconds
                    return datetime.fromtimestamp(timestamp_value, tz=timezone.utc)
            except (ValueError, OSError):
                pass
        elif isinstance(timestamp_value, datetime):
            return timestamp_value.replace(tzinfo=timezone.utc) if timestamp_value.tzinfo is None else timestamp_value
        
        return None
    
    def _process_sequential(
        self,
        data: List[Dict[str, Any]],
        processor_func: Callable[[List[Dict[str, Any]]], Any],
        **kwargs: Any
    ) -> Generator[Any, None, None]:
        """Process data sequentially in batches."""
        batch_size = self._calculate_optimal_batch_size(len(data))
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            start_time = time.time()
            result = processor_func(batch, **kwargs)
            processing_time = (time.time() - start_time) * 1000
            
            self.stats.processing_times_ms.append(processing_time)
            self.stats.batches_processed += 1
            
            # Monitor memory and performance
            self._monitor_performance(processing_time)
            
            yield result
            
            # Cleanup and memory management
            if self.config.batch_config.memory_management == MemoryManagement.CONSERVATIVE:
                self._cleanup_memory()
    
    def _process_grouped_by_counterparty(
        self,
        data: List[Dict[str, Any]],
        processor_func: Callable[[List[Dict[str, Any]]], Any],
        **kwargs: Any
    ) -> Generator[Any, None, None]:
        """Process data grouped by counterparty."""
        # Group by counterparty
        counterparty_groups: Dict[str, List[Dict[str, Any]]] = {}
        
        for record in data:
            counterparty = record.get('counterparty', record.get('counterparty_id', 'unknown'))
            if counterparty not in counterparty_groups:
                counterparty_groups[counterparty] = []
            counterparty_groups[counterparty].append(record)
        
        # Process each group
        for counterparty, group_data in counterparty_groups.items():
            if self.config.enable_progress:
                print(f"🏢 Processing counterparty: {counterparty} ({len(group_data)} records)")
            
            # Apply batching within the group
            yield from self._process_sequential(group_data, processor_func, **kwargs)
    
    def _process_temporal_chunks(
        self,
        data: List[Dict[str, Any]],
        processor_func: Callable[[List[Dict[str, Any]]], Any],
        **kwargs: Any
    ) -> Generator[Any, None, None]:
        """Process data in temporal chunks (e.g., monthly or yearly)."""
        if not self.config.time_filter:
            # Fall back to sequential processing
            yield from self._process_sequential(data, processor_func, **kwargs)
            return
        
        # Group by time periods (monthly chunks)
        temporal_groups: Dict[str, List[Dict[str, Any]]] = {}
        
        for record in data:
            timestamp = self._extract_timestamp(record)
            if timestamp:
                # Group by year-month
                key = f"{timestamp.year}-{timestamp.month:02d}"
                if key not in temporal_groups:
                    temporal_groups[key] = []
                temporal_groups[key].append(record)
        
        # Process chunks in chronological order
        for period in sorted(temporal_groups.keys()):
            group_data = temporal_groups[period]
            if self.config.enable_progress:
                print(f"📅 Processing period: {period} ({len(group_data)} records)")
            
            yield from self._process_sequential(group_data, processor_func, **kwargs)
    
    def _process_streaming(
        self,
        data: List[Dict[str, Any]],
        processor_func: Callable[[List[Dict[str, Any]]], Any],
        **kwargs: Any
    ) -> Generator[Any, None, None]:
        """Process data one record at a time."""
        for record in data:
            start_time = time.time()
            result = processor_func([record], **kwargs)
            processing_time = (time.time() - start_time) * 1000
            
            self.stats.processing_times_ms.append(processing_time)
            self.stats.batches_processed += 1
            
            yield result
    
    def _process_large_file_chunked(
        self,
        file_path: Path,
        processor_func: Callable[[List[Dict[str, Any]]], Any],
        **kwargs: Any
    ) -> Generator[Any, None, None]:
        """Process large files using chunked reading."""
        # Use datason's chunked deserialization
        chunk_size = self._calculate_optimal_batch_size()
        
        try:
            for chunk in deserialize_chunked_file(file_path, chunk_processor=None):
                if isinstance(chunk, list):
                    yield from self.process_data(chunk, processor_func, **kwargs)
                else:
                    # Single record
                    yield from self.process_data([chunk], processor_func, **kwargs)
        except Exception as e:
            self.stats.errors.append(f"Error processing chunked file: {str(e)}")
            raise
    
    def _process_file_in_memory(
        self,
        file_path: Path,
        processor_func: Callable[[List[Dict[str, Any]]], Any],
        **kwargs: Any
    ) -> Generator[Any, None, None]:
        """Process file by loading it entirely into memory."""
        try:
            # Read the entire file
            with open(file_path, 'r') as f:
                import json
                if file_path.suffix.lower() == '.jsonl':
                    data = [json.loads(line) for line in f if line.strip()]
                else:
                    data = json.load(f)
            
            if not isinstance(data, list):
                data = [data]
            
            yield from self.process_data(data, processor_func, **kwargs)
            
        except Exception as e:
            self.stats.errors.append(f"Error processing file in memory: {str(e)}")
            raise
    
    def _calculate_optimal_batch_size(self, data_size: Optional[int] = None) -> int:
        """Calculate optimal batch size based on available memory and data size."""
        base_size = self.config.batch_config.batch_size
        
        if not self.config.batch_config.dynamic_sizing:
            return base_size
        
        # Get available memory if psutil is available
        if psutil is not None:
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        else:
            # Default to conservative sizing if psutil not available
            available_memory_mb = 1024
        
        # Adjust batch size based on available memory
        if available_memory_mb < 512:  # Low memory
            batch_size = max(self.config.batch_config.min_batch_size, base_size // 4)
        elif available_memory_mb < 1024:  # Medium memory
            batch_size = base_size // 2
        elif available_memory_mb > 4096:  # High memory
            batch_size = min(self.config.batch_config.max_batch_size, base_size * 2)
        else:
            batch_size = base_size
        
        # Consider data size
        if data_size and data_size < batch_size:
            batch_size = max(self.config.batch_config.min_batch_size, data_size)
        
        return batch_size
    
    def _monitor_performance(self, processing_time_ms: float) -> None:
        """Monitor performance and issue warnings if needed."""
        if not self.config.enable_monitoring:
            return
        
        # Check processing time
        if processing_time_ms > self.config.slow_processing_threshold_ms:
            warning = f"Slow processing detected: {processing_time_ms:.1f}ms per batch"
            self.stats.warnings.append(warning)
            if self.config.enable_progress:
                print(f"⚠️ {warning}")
        
        # Check memory usage if psutil is available
        if self._process is not None:
            current_memory_mb = self._process.memory_info().rss / (1024 * 1024)
            self.stats.memory_current_mb = current_memory_mb
            
            if current_memory_mb > self.stats.memory_peak_mb:
                self.stats.memory_peak_mb = current_memory_mb
            
            if current_memory_mb > self.config.memory_warning_threshold_mb:
                warning = f"High memory usage: {current_memory_mb:.1f}MB"
                self.stats.warnings.append(warning)
                if self.config.enable_progress:
                    print(f"⚠️ {warning}")
    
    def _cleanup_memory(self) -> None:
        """Perform memory cleanup."""
        import gc
        gc.collect()
        
        # Clear caches if they get too large
        if len(self._type_cache) > 1000:
            self._type_cache.clear()
        if len(self._string_cache) > 500:
            self._string_cache.clear()
    
    def _print_final_stats(self) -> None:
        """Print final processing statistics."""
        print("\n" + "=" * 60)
        print("📊 PROCESSING STATISTICS")
        print("=" * 60)
        print(f"📝 Total records: {self.stats.total_records:,}")
        print(f"✅ Processed: {self.stats.processed_records:,}")
        print(f"🕒 Filtered by time: {self.stats.filtered_records:,}")
        print(f"⏭️ Skipped: {self.stats.skipped_records:,}")
        print(f"📦 Batches processed: {self.stats.batches_processed:,}")
        print(f"⏱️ Total time: {self.stats.total_time_seconds:.2f}s")
        print(f"🚀 Processing rate: {self.stats.records_per_second:.1f} records/sec")
        print(f"📈 Average batch time: {self.stats.average_processing_time_ms:.1f}ms")
        print(f"💾 Peak memory usage: {self.stats.memory_peak_mb:.1f}MB")
        
        if self.stats.warnings:
            print(f"\n⚠️ Warnings ({len(self.stats.warnings)}):")
            for warning in self.stats.warnings[-5:]:  # Show last 5 warnings
                print(f"  • {warning}")
        
        if self.stats.errors:
            print(f"\n❌ Errors ({len(self.stats.errors)}):")
            for error in self.stats.errors[-3:]:  # Show last 3 errors
                print(f"  • {error}")


# Convenience functions for common use cases

def create_financial_processor_config(years_back: int = 5) -> LargeDataConfig:
    """Create a configuration optimized for financial transaction processing.
    
    Args:
        years_back: Number of years to look back from the latest transaction
        
    Returns:
        Optimized configuration for financial data processing
    """
    return LargeDataConfig(
        time_filter=TimeFilterConfig(
            years_back=years_back,
            timestamp_field="transaction_date",
            fallback_timestamp_fields=[
                "date", "created_at", "processed_at", "timestamp",
                "settlement_date", "trade_date"
            ],
            sort_by_timestamp=True,
            keep_records_without_timestamp=False
        ),
        batch_config=BatchConfig(
            batch_size=500,  # Smaller batches for financial data precision
            strategy=ProcessingStrategy.GROUPED_BY_COUNTERPARTY,
            memory_management=MemoryManagement.BALANCED
        ),
        serialization_config=None,  # Will use performance config
        max_file_size_mb=50,  # Conservative for financial data
        enable_progress=True,
        enable_monitoring=True,
        slow_processing_threshold_ms=500,  # Stricter for financial processing
        memory_warning_threshold_mb=512,   # Conservative memory usage
    )


def create_time_series_processor_config(years_back: int = 10) -> LargeDataConfig:
    """Create a configuration optimized for time series analysis.
    
    Args:
        years_back: Number of years to look back from the latest data point
        
    Returns:
        Optimized configuration for time series processing
    """
    return LargeDataConfig(
        time_filter=TimeFilterConfig(
            years_back=years_back,
            timestamp_field="timestamp",
            sort_by_timestamp=True,
            keep_records_without_timestamp=False
        ),
        batch_config=BatchConfig(
            batch_size=2000,  # Larger batches for time series
            strategy=ProcessingStrategy.TEMPORAL_CHUNKS,
            memory_management=MemoryManagement.AGGRESSIVE
        ),
        max_file_size_mb=200,  # Can handle larger files for time series
        enable_progress=True,
        enable_monitoring=True,
    )


def create_streaming_processor_config() -> LargeDataConfig:
    """Create a configuration optimized for real-time streaming processing.
    
    Returns:
        Optimized configuration for streaming data processing
    """
    return LargeDataConfig(
        time_filter=None,  # No time filtering for streaming
        batch_config=BatchConfig(
            batch_size=1,  # Process one record at a time
            strategy=ProcessingStrategy.STREAMING,
            memory_management=MemoryManagement.CONSERVATIVE
        ),
        max_file_size_mb=10,  # Small files for streaming
        enable_progress=False,  # Disable progress for streaming
        enable_monitoring=True,
        slow_processing_threshold_ms=100,  # Very strict for streaming
    )