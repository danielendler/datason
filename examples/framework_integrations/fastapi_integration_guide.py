#!/usr/bin/env python3
"""
FastAPI Integration Guide with Datason

This comprehensive example demonstrates real-world FastAPI integration patterns
with datason, focusing on UUID compatibility, Pydantic models, and developer
experience best practices.

Run with: uvicorn fastapi_integration_guide:app --reload
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import datason
from datason.config import SerializationConfig, get_api_config

# =============================================================================
# SETUP & CONFIGURATION
# =============================================================================

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Datason FastAPI Integration",
    description="Real-world example of datason integration with FastAPI and Pydantic",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global datason configuration for API compatibility
API_CONFIG = get_api_config()

# Custom configuration for specific use cases
STRICT_API_CONFIG = SerializationConfig(
    uuid_format="string",  # Keep UUIDs as strings
    parse_uuids=False,  # Don't auto-convert to UUID objects
    date_format="ISO",  # ISO 8601 format
    sort_keys=True,  # Consistent ordering
    ensure_ascii=True,  # Safe for all HTTP clients
    max_depth=10,  # Security: prevent deep nesting
    max_size=1_000_000,  # Security: 1MB payload limit
)

# =============================================================================
# PYDANTIC MODELS (API CONTRACTS)
# =============================================================================


class UserBase(BaseModel):
    """Base user model with string UUIDs for API compatibility."""

    email: str = Field(..., example="user@example.com")
    name: str = Field(..., example="John Doe")
    profile: Dict[str, Any] = Field(default_factory=dict)


class UserCreate(UserBase):
    """Model for creating new users."""

    password: str = Field(..., min_length=8)


class User(UserBase):
    """Complete user model with generated fields."""

    id: str = Field(..., description="UUID as string", example="12345678-1234-5678-9012-123456789abc")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    is_active: bool = Field(default=True)

    class Config:
        # Pydantic configuration for better JSON handling
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class UserUpdate(BaseModel):
    """Model for updating users."""

    email: Optional[str] = None
    name: Optional[str] = None
    profile: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class SessionInfo(BaseModel):
    """Session information with multiple UUIDs."""

    session_id: str = Field(..., description="Session UUID as string")
    user_id: str = Field(..., description="User UUID as string")
    device_id: str = Field(..., description="Device UUID as string")
    created_at: datetime
    expires_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class APIResponse(BaseModel):
    """Standard API response wrapper."""

    success: bool = True
    data: Any = None
    message: str = "Success"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# SIMULATED DATABASE LAYER
# =============================================================================


class MockDatabase:
    """Mock database that returns data with string UUIDs (common pattern)."""

    def __init__(self):
        # Simulate database with string UUIDs (PostgreSQL/MySQL common pattern)
        self.users = {
            "12345678-1234-5678-9012-123456789abc": {
                "id": "12345678-1234-5678-9012-123456789abc",
                "email": "john@example.com",
                "name": "John Doe",
                "created_at": "2023-01-01T12:00:00Z",  # ISO string from DB
                "updated_at": "2023-06-15T14:30:45Z",
                "is_active": True,
                "profile": {
                    "theme": "dark",
                    "language": "en",
                    "preferences": {"notifications": True, "newsletter": False},
                },
            },
            "87654321-4321-8765-2109-cba987654321": {
                "id": "87654321-4321-8765-2109-cba987654321",
                "email": "jane@example.com",
                "name": "Jane Smith",
                "created_at": "2023-02-15T08:30:00Z",
                "updated_at": "2023-06-15T14:30:45Z",
                "is_active": True,
                "profile": {"theme": "light", "language": "es"},
            },
        }

        self.sessions = {
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee": {
                "session_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                "user_id": "12345678-1234-5678-9012-123456789abc",
                "device_id": "ffffffff-eeee-dddd-cccc-bbbbbbbbbbbb",
                "created_at": "2023-06-15T14:00:00Z",
                "expires_at": "2023-06-16T14:00:00Z",
                "metadata": {"ip_address": "192.168.1.1", "user_agent": "Mozilla/5.0...", "location": "New York, NY"},
            }
        }

    def get_user(self, user_id: str) -> Optional[Dict]:
        return self.users.get(user_id)

    def get_all_users(self) -> List[Dict]:
        return list(self.users.values())

    def create_user(self, user_data: Dict) -> Dict:
        user_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        new_user = {"id": user_id, "created_at": now, "updated_at": now, "is_active": True, **user_data}

        self.users[user_id] = new_user
        return new_user

    def update_user(self, user_id: str, updates: Dict) -> Optional[Dict]:
        if user_id not in self.users:
            return None

        user = self.users[user_id].copy()
        user.update(updates)
        user["updated_at"] = datetime.now(timezone.utc).isoformat()

        self.users[user_id] = user
        return user

    def get_session(self, session_id: str) -> Optional[Dict]:
        return self.sessions.get(session_id)


# Global database instance
db = MockDatabase()

# =============================================================================
# DATASON PROCESSING HELPERS
# =============================================================================


def process_api_data(data: Any, config: SerializationConfig = None) -> Any:
    """
    Process data with datason for API compatibility.

    This helper ensures consistent processing across all endpoints.
    """
    if config is None:
        config = API_CONFIG

    try:
        # Auto-deserialize to handle complex types intelligently
        processed = datason.auto_deserialize(data, config=config)
        logger.info(f"Processed data with config: {config.uuid_format} UUIDs")
        return processed
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        # Fallback to original data if processing fails
        return data


def serialize_api_response(data: Any, config: SerializationConfig = None) -> Any:
    """
    Serialize data for API responses.

    Ensures consistent output format for all API responses.
    """
    if config is None:
        config = API_CONFIG

    try:
        serialized = datason.serialize(data, config=config)
        return serialized
    except Exception as e:
        logger.error(f"Response serialization failed: {e}")
        return data


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================


async def get_database():
    """Dependency to get database instance."""
    return db


async def get_processing_config():
    """Dependency to get datason processing configuration."""
    return API_CONFIG


# =============================================================================
# API ENDPOINTS
# =============================================================================


@app.get("/", response_model=APIResponse)
async def root():
    """Root endpoint with datason processing demonstration."""

    demo_data = {
        "message": "Welcome to Datason FastAPI Integration!",
        "features": [
            "UUID string compatibility with Pydantic",
            "Intelligent datetime handling",
            "Nested data processing",
            "High-performance serialization",
        ],
        "sample_uuids": ["12345678-1234-5678-9012-123456789abc", "87654321-4321-8765-2109-cba987654321"],
        "timestamp": datetime.now(timezone.utc),
    }

    # Process with datason
    processed_data = process_api_data(demo_data)

    return APIResponse(data=processed_data, message="API is running with datason integration")


@app.get("/users/", response_model=List[User])
async def list_users(database=Depends(get_database), config=Depends(get_processing_config)):
    """
    List all users with datason processing.

    Demonstrates:
    - Batch processing of multiple records
    - UUID string preservation
    - Datetime conversion from ISO strings
    """

    # Get raw data from database (simulates real DB queries)
    raw_users = database.get_all_users()

    # Process with datason for consistent API output
    processed_users = process_api_data(raw_users, config)

    # Convert to Pydantic models (this now works perfectly!)
    users = [User(**user_data) for user_data in processed_users]

    logger.info(f"Listed {len(users)} users with datason processing")
    return users


@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: str, database=Depends(get_database), config=Depends(get_processing_config)):
    """
    Get specific user by ID.

    Demonstrates:
    - UUID parameter handling
    - Single record processing
    - Error handling with processed data
    """

    # Validate UUID format (optional but recommended)
    try:
        uuid.UUID(user_id)  # Validates UUID format
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid UUID format") from e

    # Get raw data from database
    raw_user = database.get_user(user_id)
    if not raw_user:
        raise HTTPException(status_code=404, detail="User not found")

    # Process with datason
    processed_user = process_api_data(raw_user, config)

    # Return as Pydantic model
    return User(**processed_user)


@app.post("/users/", response_model=User, status_code=201)
async def create_user(user_data: UserCreate, database=Depends(get_database), config=Depends(get_processing_config)):
    """
    Create new user.

    Demonstrates:
    - Processing incoming Pydantic data
    - Database creation with UUID generation
    - Response processing
    """

    # Convert Pydantic model to dict
    user_dict = user_data.dict()

    # Remove password for storage (in real app, hash it)
    _ = user_dict.pop("password")  # Use underscore to indicate intentionally unused
    logger.info(f"Creating user with email: {user_dict['email']}")

    # Create in database (returns data with string UUIDs)
    raw_user = database.create_user(user_dict)

    # Process with datason
    processed_user = process_api_data(raw_user, config)

    # Return as Pydantic model
    return User(**processed_user)


@app.put("/users/{user_id}", response_model=User)
async def update_user(
    user_id: str, updates: UserUpdate, database=Depends(get_database), config=Depends(get_processing_config)
):
    """
    Update existing user.

    Demonstrates:
    - Partial updates
    - UUID preservation through update process
    """

    # Validate UUID
    try:
        uuid.UUID(user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid UUID format") from e

    # Get update data (only non-None fields)
    update_data = {k: v for k, v in updates.dict().items() if v is not None}

    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    # Update in database
    raw_user = database.update_user(user_id, update_data)
    if not raw_user:
        raise HTTPException(status_code=404, detail="User not found")

    # Process with datason
    processed_user = process_api_data(raw_user, config)

    return User(**processed_user)


@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str, database=Depends(get_database), config=Depends(get_processing_config)):
    """
    Get session information.

    Demonstrates:
    - Multiple UUID fields in single response
    - Nested metadata processing
    - Complex data structure handling
    """

    # Validate session UUID
    try:
        uuid.UUID(session_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid session UUID format") from e

    # Get raw session data
    raw_session = database.get_session(session_id)
    if not raw_session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Process with datason - handles ALL UUIDs consistently
    processed_session = process_api_data(raw_session, config)

    return SessionInfo(**processed_session)


@app.get("/demo/complex-data", response_model=APIResponse)
async def demo_complex_data(config=Depends(get_processing_config)):
    """
    Demonstrate datason with complex nested data structures.

    Shows:
    - Multiple nested UUIDs
    - Mixed data types
    - Performance with larger datasets
    """

    complex_data = {
        "user_id": "12345678-1234-5678-9012-123456789abc",
        "workspace": {
            "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "owner_id": "87654321-4321-8765-2109-cba987654321",
            "settings": {
                "theme_id": "ffffffff-eeee-dddd-cccc-bbbbbbbbbbbb",
                "created_at": "2023-01-01T12:00:00Z",
                "updated_at": "2023-06-15T14:30:45Z",
            },
        },
        "collaborators": [
            {"user_id": "11111111-2222-3333-4444-555555555555", "role": "editor", "invited_at": "2023-02-01T10:00:00Z"},
            {"user_id": "66666666-7777-8888-9999-aaaaaaaaaaaa", "role": "viewer", "invited_at": "2023-03-01T15:30:00Z"},
        ],
        "recent_activity": [
            {
                "id": "activity-1-uuid-string-here",
                "user_id": "12345678-1234-5678-9012-123456789abc",
                "action": "document_created",
                "timestamp": "2023-06-15T14:00:00Z",
                "metadata": {
                    "document_id": "doc-uuid-12345678-1234-5678",
                    "location": {"lat": 40.7128, "lng": -74.0060},
                },
            }
        ],
    }

    # Process complex nested structure
    processed_data = process_api_data(complex_data, config)

    return APIResponse(data=processed_data, message="Complex nested data processed successfully")


@app.get("/demo/performance", response_model=APIResponse)
async def demo_performance(count: int = 100, config=Depends(get_processing_config)):
    """
    Performance demonstration with large datasets.

    Shows datason performance with configurable dataset sizes.
    """
    import time

    if count > 10000:
        raise HTTPException(status_code=400, detail="Count too large (max 10000)")

    # Generate large dataset
    large_dataset = [
        {
            "id": f"{i:08x}-1234-5678-9012-123456789abc",
            "user_id": f"{i:08x}-4321-8765-2109-cba987654321",
            "created_at": "2023-01-01T12:00:00Z",
            "data": {"value": i, "text": f"Item {i}"},
        }
        for i in range(count)
    ]

    # Measure processing time
    start_time = time.perf_counter()
    processed_dataset = process_api_data(large_dataset, config)
    processing_time = time.perf_counter() - start_time

    return APIResponse(
        data={
            "count": count,
            "processing_time_seconds": round(processing_time, 4),
            "records_per_second": round(count / processing_time, 2),
            "sample_record": processed_dataset[0] if processed_dataset else None,
        },
        message=f"Processed {count} records in {processing_time:.4f}s",
    )


# =============================================================================
# MIDDLEWARE EXAMPLE
# =============================================================================


@app.middleware("http")
async def datason_middleware(request: Request, call_next):
    """
    Optional middleware for automatic datason processing.

    This demonstrates how to automatically process all requests/responses.
    """

    # Process request (if needed)
    if request.method in ["POST", "PUT", "PATCH"] and request.headers.get("content-type") == "application/json":
        # In a real implementation, you might want to process request body here
        pass

    # Process the request
    response = await call_next(request)

    # Add processing metadata to response headers
    response.headers["X-Datason-Processed"] = "true"
    response.headers["X-Datason-Config"] = "api-compatible"

    return response


# =============================================================================
# HEALTH CHECK & UTILITY ENDPOINTS
# =============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "datason_config": "api-compatible",
        "uuid_format": "string",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/config-info")
async def config_info(config=Depends(get_processing_config)):
    """Get current datason configuration information."""

    config_info = {
        "uuid_format": getattr(config, "uuid_format", "object"),
        "parse_uuids": getattr(config, "parse_uuids", True),
        "date_format": getattr(config, "date_format", "ISO"),
        "sort_keys": getattr(config, "sort_keys", False),
        "ensure_ascii": getattr(config, "ensure_ascii", False),
        "max_depth": getattr(config, "max_depth", None),
        "max_size": getattr(config, "max_size", None),
    }

    return APIResponse(data=config_info, message="Current datason configuration")


# =============================================================================
# STARTUP EVENT
# =============================================================================


@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info("FastAPI app starting up with datason integration")
    logger.info(f"API Config - UUID format: {getattr(API_CONFIG, 'uuid_format', 'unknown')}")
    logger.info(f"API Config - Parse UUIDs: {getattr(API_CONFIG, 'parse_uuids', 'unknown')}")


# =============================================================================
# MAIN (FOR TESTING)
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting FastAPI with Datason Integration")
    print("📖 Visit http://localhost:8000/docs for interactive API docs")
    print("🎯 Key endpoints:")
    print("   • GET  /users/           - List users with UUID processing")
    print("   • GET  /users/{id}       - Get user by UUID")
    print("   • POST /users/           - Create user with validation")
    print("   • GET  /sessions/{id}    - Complex nested UUID handling")
    print("   • GET  /demo/complex-data - Nested data structures")
    print("   • GET  /demo/performance - Performance testing")
    print("   • GET  /config-info      - View current configuration")

    uvicorn.run("fastapi_integration_guide:app", host="0.0.0.0", port=8000, reload=True, log_level="info")  # nosec B104 - Demo binding
