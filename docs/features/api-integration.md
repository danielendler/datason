# API Integration & Framework Compatibility

Datason provides seamless integration with modern Python web frameworks like **FastAPI**, **Django**, **Flask**, and validation libraries like **Pydantic**. This guide focuses on real-world developer experience and practical integration patterns.

## 🚀 Quick Start: FastAPI + Pydantic

The most common integration challenge: **UUID compatibility with Pydantic models**.

### ❌ The Problem

```python
import datason
from pydantic import BaseModel

# Data from your database/API
data = {"user_id": "ea82f3dd-d770-41b9-9706-69cd3070b4f5", "name": "John"}

# Default datason behavior converts UUID strings to UUID objects
result = datason.auto_deserialize(data)
# result = {"user_id": UUID('ea82f3dd-d770-41b9-9706-69cd3070b4f5'), "name": "John"}

# But your Pydantic model expects strings
class User(BaseModel):
    user_id: str  # ❌ This fails! UUID object != string
    name: str

user = User(**result)  # ValidationError: str type expected
```

### ✅ The Solution

```python
import datason
from datason.config import get_api_config
from pydantic import BaseModel

# Use API configuration for Pydantic compatibility
api_config = get_api_config()
result = datason.auto_deserialize(data, config=api_config)
# result = {"user_id": "ea82f3dd-d770-41b9-9706-69cd3070b4f5", "name": "John"}

class User(BaseModel):
    user_id: str  # ✅ Works perfectly!
    name: str

user = User(**result)  # Success! 🎉
```

## 🎯 Framework-Specific Examples

### FastAPI: Complete Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
import datason
from datason.config import get_api_config

app = FastAPI()

# Your Pydantic models
class User(BaseModel):
    id: str  # UUID as string
    email: str
    created_at: datetime
    profile: Dict[str, Any]

class CreateUserRequest(BaseModel):
    email: str
    profile: Dict[str, Any]

# Set up API configuration once
API_CONFIG = get_api_config()

@app.post("/users/", response_model=User)
async def create_user(request: CreateUserRequest):
    # Simulate database response (common real-world pattern)
    db_data = {
        "id": "12345678-1234-5678-9012-123456789abc",
        "email": request.email,
        "created_at": "2023-01-01T12:00:00Z",  # ISO string from DB
        "profile": request.profile
    }

    # Process with datason - keeps UUIDs as strings, converts dates
    processed = datason.auto_deserialize(db_data, config=API_CONFIG)

    # Now Pydantic validation works perfectly!
    return User(**processed)

@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: str):
    # Simulate database lookup
    db_result = fetch_user_from_db(user_id)  # Returns dict with string UUIDs

    # Process and return
    processed = datason.auto_deserialize(db_result, config=API_CONFIG)
    return User(**processed)

@app.get("/users/", response_model=List[User])
async def list_users():
    # Large dataset processing
    db_results = fetch_all_users()  # List of dicts

    # Process entire list efficiently
    processed = datason.auto_deserialize(db_results, config=API_CONFIG)
    return [User(**user) for user in processed]
```

### Django: API Views & Models

```python
from django.http import JsonResponse
from django.views import View
from django.forms.models import model_to_dict
import datason
from datason.config import get_api_config
from .models import User

class UserAPIView(View):

    def __init__(self):
        super().__init__()
        self.api_config = get_api_config()

    def get(self, request, user_id):
        """Get user data with proper serialization."""
        try:
            user = User.objects.get(id=user_id)

            # Convert Django model to dict
            user_data = model_to_dict(user)

            # Process with datason for consistent API responses
            processed = datason.serialize(user_data, config=self.api_config)

            return JsonResponse(processed)
        except User.DoesNotExist:
            return JsonResponse({"error": "User not found"}, status=404)

    def post(self, request):
        """Create user from JSON data."""
        import json

        # Parse request data
        data = json.loads(request.body)

        # Process incoming data for consistency
        processed = datason.auto_deserialize(data, config=self.api_config)

        # Create user (processed data is now Django-compatible)
        user = User.objects.create(**processed)

        # Return processed response
        response_data = model_to_dict(user)
        serialized = datason.serialize(response_data, config=self.api_config)

        return JsonResponse(serialized, status=201)
```

### Flask: API Endpoints

```python
from flask import Flask, request, jsonify
from datetime import datetime
import datason
from datason.config import get_api_config

app = Flask(__name__)

# Global API configuration
API_CONFIG = get_api_config()

@app.route('/api/users/<user_id>', methods=['GET'])
def get_user(user_id):
    """Get user with proper UUID handling."""

    # Simulate database response
    db_user = {
        "id": user_id,  # Already a string UUID
        "email": "user@example.com",
        "created_at": "2023-01-01T12:00:00Z",
        "preferences": {"theme": "dark", "lang": "en"}
    }

    # Process for API response
    response_data = datason.auto_deserialize(db_user, config=API_CONFIG)

    return jsonify(response_data)

@app.route('/api/users/', methods=['POST'])
def create_user():
    """Create user with data validation."""

    data = request.json

    # Validate and process input data
    processed_input = datason.auto_deserialize(data, config=API_CONFIG)

    # Add server-generated fields
    processed_input.update({
        "id": "12345678-1234-5678-9012-123456789abc",
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    })

    # Serialize for database storage and API response
    response_data = datason.serialize(processed_input, config=API_CONFIG)

    return jsonify(response_data), 201

# Middleware for automatic request/response processing
@app.before_request
def process_request():
    """Automatically process JSON requests."""
    if request.is_json:
        # Store processed data for route handlers
        request.processed_json = datason.auto_deserialize(
            request.json,
            config=API_CONFIG
        )

@app.after_request
def process_response(response):
    """Automatically process JSON responses."""
    if response.is_json:
        # Ensure consistent serialization
        import json
        data = json.loads(response.data)
        processed = datason.serialize(data, config=API_CONFIG)
        response.data = json.dumps(processed)

    return response
```

## 🔧 Configuration Strategies

### Quick Configuration Selector

```python
# Choose your configuration based on use case:

# 🚀 Web APIs (FastAPI, Django REST, Flask APIs)
from datason.config import get_api_config
config = get_api_config()  # UUIDs as strings, ISO dates

# 🔬 ML/Data Processing  
from datason.config import get_ml_config
config = get_ml_config()   # UUIDs as objects, rich types

# ⚡ High Performance
from datason.config import get_performance_config  
config = get_performance_config()  # Minimal processing

# 🛡️ Security Focused
from datason.config import get_security_config
config = get_security_config()  # Size limits, depth limits
```

### Custom Configuration for Specific Needs

```python
from datason.config import SerializationConfig, DateFormat, NanHandling

# Custom API configuration with specific requirements
custom_api_config = SerializationConfig(
    # UUID handling for API compatibility
    uuid_format="string",      # Keep UUIDs as strings
    parse_uuids=False,         # Don't auto-convert to UUID objects

    # Date/time handling  
    date_format=DateFormat.ISO,  # ISO 8601 format

    # JSON formatting for APIs
    sort_keys=True,            # Consistent key ordering
    ensure_ascii=True,         # Safe for all HTTP clients

    # Data handling
    nan_handling=NanHandling.NULL,  # Convert NaN to null
    preserve_decimals=True,    # Keep decimal precision

    # Performance & security
    max_depth=10,              # Prevent deeply nested attacks
    max_size=1_000_000        # 1MB limit for API payloads
)
```

## 🎨 Advanced Integration Patterns

### Middleware Integration

```python
# FastAPI Middleware Example
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import datason
from datason.config import get_api_config

class DatasonMiddleware(BaseHTTPMiddleware):
    """Automatic datason processing for all requests/responses."""

    def __init__(self, app, config=None):
        super().__init__(app)
        self.config = config or get_api_config()

    async def dispatch(self, request: Request, call_next):
        # Process request data if JSON
        if request.headers.get("content-type") == "application/json":
            body = await request.body()
            if body:
                import json
                data = json.loads(body)
                processed = datason.auto_deserialize(data, config=self.config)
                # Store processed data for route handlers
                request.state.processed_data = processed

        response = await call_next(request)

        # Process response data
        if response.headers.get("content-type") == "application/json":
            # Response processing would go here
            pass

        return response

# Use the middleware
app = FastAPI()
app.add_middleware(DatasonMiddleware)
```

### Database Integration Helpers

```python
import datason
from datason.config import get_api_config

class DatasonMixin:
    """Mixin for models with datason serialization."""

    @classmethod
    def get_config(cls):
        """Override to customize configuration per model."""
        return get_api_config()

    def to_datason(self):
        """Serialize model instance."""
        data = self.__dict__.copy()
        # Remove SQLAlchemy internal attributes
        data = {k: v for k, v in data.items() if not k.startswith('_')}
        return datason.serialize(data, config=self.get_config())

    @classmethod
    def from_datason(cls, data):
        """Create instance from datason data."""
        processed = datason.auto_deserialize(data, config=cls.get_config())
        return cls(**processed)

# Usage with SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base, DatasonMixin):
    __tablename__ = 'users'

    id = Column(String, primary_key=True)  # UUID as string
    email = Column(String)
    created_at = Column(DateTime)

# Now you can:
user = User.query.first()
serialized = user.to_datason()  # Ready for API response

# And:
new_user = User.from_datason(request_data)  # From API request
```

## 🚨 Common Pitfalls & Solutions

### Pitfall 1: Inconsistent Configuration

```python
# ❌ Don't mix configurations
result1 = datason.auto_deserialize(data)  # Default config
result2 = datason.auto_deserialize(data, config=get_api_config())  # API config
# UUIDs will be different types!

# ✅ Use consistent configuration
API_CONFIG = get_api_config()
result1 = datason.auto_deserialize(data1, config=API_CONFIG)
result2 = datason.auto_deserialize(data2, config=API_CONFIG)
```

### Pitfall 2: Forgetting Nested UUIDs

```python
# Complex nested data with multiple UUIDs
data = {
    "user_id": "12345678-1234-5678-9012-123456789abc",
    "session": {
        "id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "device_id": "ffffffff-eeee-dddd-cccc-bbbbbbbbbbbb"
    },
    "related_users": [
        {"id": "11111111-2222-3333-4444-555555555555"},
        {"id": "66666666-7777-8888-9999-aaaaaaaaaaaa"}
    ]
}

# ✅ API config handles ALL nested UUIDs consistently
processed = datason.auto_deserialize(data, config=get_api_config())
# All UUIDs remain as strings, regardless of nesting level
```

### Pitfall 3: Performance with Large Datasets

```python
# ❌ Processing large datasets item by item
users = []
for user_data in large_dataset:
    processed = datason.auto_deserialize(user_data, config=api_config)
    users.append(processed)

# ✅ Process entire dataset at once
processed_users = datason.auto_deserialize(large_dataset, config=api_config)
```

## 📊 Performance Guidelines

### API Response Times

- **Small responses** (< 1KB): ~0.001s overhead
- **Medium responses** (1-100KB): ~0.01s overhead  
- **Large responses** (100KB-1MB): ~0.1s overhead
- **Batch operations** (1000+ items): Use list processing

### Memory Usage

- **String UUIDs**: ~40% less memory than UUID objects
- **API config**: Optimized for minimal memory overhead
- **Large datasets**: Consider chunked processing for > 10MB

## 🔗 Integration Checklist

### For New Projects

- [ ] Choose appropriate configuration preset
- [ ] Set up consistent config across your app
- [ ] Define your API models with string UUIDs
- [ ] Add datason processing to request/response pipeline
- [ ] Test with real-world data patterns

### For Existing Projects

- [ ] Identify current UUID/datetime pain points
- [ ] Add `get_api_config()` to problem areas
- [ ] Update Pydantic models if needed
- [ ] Test backward compatibility
- [ ] Roll out incrementally

## 📚 Related Documentation

- [Configuration System](configuration/index.md) - Complete configuration reference
- [Type Handling](advanced-types/index.md) - How datason handles different Python types
- [Performance Guide](performance/index.md) - Optimization strategies
- [Security Guide](../community/security.md) - Security considerations for web APIs

## 💡 Need Help?

- **FastAPI specific questions**: Check our [FastAPI examples](../user-guide/examples/index.md)
- **Django integration**: See [Django integration patterns](../user-guide/examples/index.md)
- **Performance optimization**: Review [Performance Guide](performance/index.md)
- **Custom configurations**: See [Configuration System](configuration/index.md)

The UUID/Pydantic compatibility issue is now solved! 🎉 Use `get_api_config()` for instant compatibility with modern Python web frameworks.
