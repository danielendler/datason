#!/usr/bin/env python3
"""
Django Integration Guide with Datason

This comprehensive example demonstrates real-world Django integration patterns
with datason, focusing on models, views, forms, and API serialization.

Setup:
    pip install django djangorestframework
    python manage.py migrate
    python manage.py runserver
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from django import forms
from django.conf import settings
from django.contrib import admin
from django.core.management.base import BaseCommand
from django.db import models
from django.forms.models import model_to_dict
from django.http import JsonResponse
from django.urls import include, path
from django.views import View

try:
    from rest_framework import serializers, viewsets
    from rest_framework.decorators import action
    from rest_framework.response import Response

    HAS_DRF = True
except ImportError:
    HAS_DRF = False

import datason
from datason.config import SerializationConfig, get_api_config

# =============================================================================
# DJANGO SETTINGS CONFIGURATION (for standalone example)
# =============================================================================

if not settings.configured:
    settings.configure(
        DEBUG=True,
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        },
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            "django.contrib.admin",
            "django.contrib.sessions",
            "django.contrib.messages",
            "rest_framework" if HAS_DRF else None,
        ],
        SECRET_KEY="dev-secret-key-for-example",  # nosec B106 - Demo only, not production
        USE_TZ=True,
        ROOT_URLCONF=__name__,
    )

# =============================================================================
# DATASON CONFIGURATION FOR DJANGO
# =============================================================================

# API-compatible configuration for Django REST APIs
DJANGO_API_CONFIG = get_api_config()

# Django-specific configuration for form processing
DJANGO_FORM_CONFIG = SerializationConfig(
    uuid_format="string",  # Forms typically work with strings
    parse_uuids=False,  # Don't auto-convert form UUIDs
    date_format="ISO",  # Standard ISO format
    preserve_decimals=True,  # Important for financial data
    sort_keys=True,  # Consistent JSON output
)

# Configuration for database JSON fields
DJANGO_JSON_FIELD_CONFIG = SerializationConfig(
    uuid_format="string",  # Store UUIDs as strings in JSON
    parse_uuids=False,  # Prevent conversion in JSON fields
    preserve_decimals=True,  # Keep precision in JSON
    max_depth=20,  # Allow deeper nesting in JSON fields
)

# =============================================================================
# DJANGO MODELS WITH DATASON INTEGRATION
# =============================================================================


class DatasonModelMixin:
    """
    Mixin for Django models to add datason serialization capabilities.

    This provides consistent serialization across all models that use it.
    """

    @classmethod
    def get_datason_config(cls):
        """Override this to customize config per model."""
        return DJANGO_API_CONFIG

    def to_datason(self, exclude_fields: List[str] = None, config=None):
        """
        Serialize model instance to datason-processed dict.

        Args:
            exclude_fields: Fields to exclude from serialization
            config: Custom datason configuration
        """
        if config is None:
            config = self.get_datason_config()

        if exclude_fields is None:
            exclude_fields = ["password", "password_hash"]

        # Convert model to dict
        data = model_to_dict(self)

        # Remove excluded fields
        for field in exclude_fields:
            data.pop(field, None)

        # Add related field data if needed
        for field in self._meta.get_fields():
            if hasattr(field, "related_model") and hasattr(self, field.name):
                try:
                    related_obj = getattr(self, field.name)
                    if related_obj and hasattr(related_obj, "to_datason"):
                        data[f"{field.name}_data"] = related_obj.to_datason(config=config)
                except Exception:  # More specific exception handling
                    pass  # nosec B110 - Skip if relation can't be loaded (demo code)

        # Process with datason
        return datason.serialize(data, config=config)

    @classmethod
    def from_datason(cls, data: Dict[str, Any], config=None):
        """
        Create model instance from datason-processed data.

        Args:
            data: Dictionary data to process
            config: Custom datason configuration
        """
        if config is None:
            config = cls.get_datason_config()

        # Process with datason
        processed_data = datason.auto_deserialize(data, config=config)

        # Remove any non-model fields
        model_fields = {f.name for f in cls._meta.get_fields()}
        clean_data = {k: v for k, v in processed_data.items() if k in model_fields}

        return cls(**clean_data)


class User(models.Model, DatasonModelMixin):
    """
    User model with UUID primary key and datason integration.

    Demonstrates:
    - String UUID primary key (common Django pattern)
    - JSON field for profile data
    - Datetime fields
    - Datason serialization methods
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField(unique=True)
    name = models.CharField(max_length=255)
    profile = models.JSONField(default=dict, blank=True)  # Will use datason for JSON processing
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "users"
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.email})"

    def save(self, *args, **kwargs):
        """Override save to process JSON fields with datason."""
        if self.profile:
            # Process profile data with datason before saving
            self.profile = datason.serialize(self.profile, config=DJANGO_JSON_FIELD_CONFIG)
        super().save(*args, **kwargs)

    @property
    def processed_profile(self):
        """Get profile data processed with datason."""
        if not self.profile:
            return {}
        return datason.auto_deserialize(self.profile, config=DJANGO_JSON_FIELD_CONFIG)


class Organization(models.Model, DatasonModelMixin):
    """Organization model with foreign key relationships."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="owned_organizations")
    members = models.ManyToManyField(User, through="Membership", related_name="organizations")
    settings = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


class Membership(models.Model, DatasonModelMixin):
    """Through model for User-Organization relationship."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE)
    role = models.CharField(
        max_length=50,
        choices=[
            ("owner", "Owner"),
            ("admin", "Admin"),
            ("member", "Member"),
        ],
    )
    joined_at = models.DateTimeField(auto_now_add=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        unique_together = ["user", "organization"]


# =============================================================================
# DJANGO ADMIN INTEGRATION
# =============================================================================


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    """Admin interface with datason-processed display."""

    list_display = ["name", "email", "is_active", "created_at"]
    list_filter = ["is_active", "created_at"]
    search_fields = ["name", "email"]
    readonly_fields = ["id", "created_at", "updated_at", "processed_profile_display"]

    def processed_profile_display(self, obj):
        """Display processed profile data in admin."""
        if not obj.profile:
            return "No profile data"

        processed = datason.serialize(obj.profile, config=DJANGO_API_CONFIG)
        return json.dumps(processed, indent=2)

    processed_profile_display.short_description = "Processed Profile (Datason)"


# =============================================================================
# DJANGO VIEWS WITH DATASON INTEGRATION
# =============================================================================


class DatasonAPIView(View):
    """
    Base API view with datason processing.

    Provides consistent datason processing for all API endpoints.
    """

    def get_datason_config(self):
        """Get datason configuration for this view."""
        return DJANGO_API_CONFIG

    def process_request_data(self, data):
        """Process incoming request data with datason."""
        if not data:
            return data
        return datason.auto_deserialize(data, config=self.get_datason_config())

    def process_response_data(self, data):
        """Process outgoing response data with datason."""
        if not data:
            return data
        return datason.serialize(data, config=self.get_datason_config())

    def json_response(self, data, status=200):
        """Create JSON response with datason processing."""
        try:
            processed_data = self.process_response_data(data)
            return JsonResponse(processed_data, status=status, safe=False)
        except Exception:
            # Don't expose internal error details to external users
            error_response = {"error": "An internal error occurred while processing the request."}
            return JsonResponse(error_response, status=500, safe=False)


class UserListAPIView(DatasonAPIView):
    """API view for listing and creating users."""

    def get(self, request):
        """List all users with datason processing."""
        # Get all users
        users = User.objects.all()

        # Convert to list of dicts
        user_data = [user.to_datason() for user in users]

        return self.json_response(
            {"users": user_data, "count": len(user_data), "timestamp": datetime.now(timezone.utc).isoformat()}
        )

    def post(self, request):
        """Create new user with datason processing."""
        try:
            # Parse request data
            data = json.loads(request.body)

            # Process with datason
            processed_data = self.process_request_data(data)

            # Create user
            user = User.from_datason(processed_data)
            user.save()

            return self.json_response(user.to_datason(), status=201)

        except json.JSONDecodeError:
            return self.json_response({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            # Log the error internally for debugging, but don't expose details to users
            import logging

            logging.error("An error occurred while creating user: %s", e, exc_info=True)
            return self.json_response({"error": "An internal error occurred while processing the request."}, status=400)


class UserDetailAPIView(DatasonAPIView):
    """API view for user detail operations."""

    def get(self, request, user_id):
        """Get user by ID with datason processing."""
        try:
            # Django automatically handles UUID string conversion
            user = User.objects.get(id=user_id)
            return self.json_response(user.to_datason())

        except User.DoesNotExist:
            return self.json_response({"error": "User not found"}, status=404)
        except ValueError:
            return self.json_response({"error": "Invalid UUID format"}, status=400)

    def put(self, request, user_id):
        """Update user with datason processing."""
        try:
            user = User.objects.get(id=user_id)

            # Parse and process update data
            data = json.loads(request.body)
            processed_data = self.process_request_data(data)

            # Update user fields
            for field, value in processed_data.items():
                if hasattr(user, field):
                    setattr(user, field, value)

            user.save()
            return self.json_response(user.to_datason())

        except User.DoesNotExist:
            return self.json_response({"error": "User not found"}, status=404)
        except json.JSONDecodeError:
            return self.json_response({"error": "Invalid JSON"}, status=400)
        except Exception as e:
            # Log the error internally for debugging, but don't expose details to users
            import logging

            logging.error("An error occurred while updating user: %s", e, exc_info=True)
            return self.json_response({"error": "An internal error occurred while processing the request."}, status=400)


class OrganizationAPIView(DatasonAPIView):
    """API view for organizations with complex relationships."""

    def get(self, request, org_id):
        """Get organization with all related data."""
        try:
            org = Organization.objects.select_related("owner").prefetch_related("members").get(id=org_id)

            # Build comprehensive organization data
            org_data = {
                "id": str(org.id),  # Ensure UUID is string
                "name": org.name,
                "owner": org.owner.to_datason(),
                "settings": org.settings,
                "created_at": org.created_at.isoformat(),
                "updated_at": org.updated_at.isoformat(),
                "members": [
                    {
                        "user": member.to_datason(),
                        "membership": Membership.objects.get(user=member, organization=org).to_datason(),
                    }
                    for member in org.members.all()
                ],
            }

            return self.json_response(org_data)

        except Organization.DoesNotExist:
            return self.json_response({"error": "Organization not found"}, status=404)


# =============================================================================
# DJANGO REST FRAMEWORK INTEGRATION (if available)
# =============================================================================

if HAS_DRF:

    class DatasonDRFSerializer(serializers.Serializer):
        """
        Base DRF serializer with datason processing.

        Integrates datason processing into Django REST Framework.
        """

        def get_datason_config(self):
            """Get datason configuration for this serializer."""
            return DJANGO_API_CONFIG

        def to_representation(self, instance):
            """Override to add datason processing to output."""
            # Get standard DRF representation
            data = super().to_representation(instance)

            # Process with datason for consistency
            return datason.serialize(data, config=self.get_datason_config())

        def to_internal_value(self, data):
            """Override to add datason processing to input."""
            # Process input with datason first
            processed_data = datason.auto_deserialize(data, config=self.get_datason_config())

            # Then apply standard DRF validation
            return super().to_internal_value(processed_data)

    class UserDRFSerializer(DatasonDRFSerializer):
        """DRF serializer for User model."""

        id = serializers.CharField(read_only=True)  # UUID as string
        email = serializers.EmailField()
        name = serializers.CharField(max_length=255)
        profile = serializers.JSONField(default=dict)
        is_active = serializers.BooleanField(default=True)
        created_at = serializers.DateTimeField(read_only=True)
        updated_at = serializers.DateTimeField(read_only=True)

        class Meta:
            model = User
            fields = "__all__"

    class UserDRFViewSet(viewsets.ModelViewSet):
        """DRF ViewSet with datason integration."""

        queryset = User.objects.all()
        serializer_class = UserDRFSerializer

        def get_datason_config(self):
            """Get datason configuration for this viewset."""
            return DJANGO_API_CONFIG

        @action(detail=True, methods=["post"])
        def update_profile(self, request, pk=None):
            """Custom action to update user profile with datason processing."""
            user = self.get_object()

            # Process profile data with datason
            profile_data = datason.auto_deserialize(request.data.get("profile", {}), config=self.get_datason_config())

            user.profile.update(profile_data)
            user.save()

            return Response(user.to_datason())

        @action(detail=False, methods=["get"])
        def export_all(self, request):
            """Export all users with datason processing."""
            users = self.get_queryset()

            # Process entire dataset with datason
            user_data = [user.to_datason() for user in users]
            processed_data = datason.serialize(user_data, config=self.get_datason_config())

            try:
                return Response(
                    {
                        "users": processed_data,
                        "count": len(processed_data),
                        "exported_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
            except Exception:
                # Don't expose internal error details to external users
                error_response = {"error": "An internal error occurred while processing the request."}
                return Response(error_response, status=500)

# =============================================================================
# DJANGO FORMS WITH DATASON INTEGRATION
# =============================================================================


class DatasonFormMixin:
    """
    Mixin for Django forms to add datason processing.
    """

    def get_datason_config(self):
        """Get datason configuration for form processing."""
        return DJANGO_FORM_CONFIG

    def clean(self):
        """Override clean to add datason processing."""
        cleaned_data = super().clean()

        # Process form data with datason
        processed_data = datason.auto_deserialize(cleaned_data, config=self.get_datason_config())

        return processed_data


class UserForm(forms.ModelForm, DatasonFormMixin):
    """Form for User model with datason processing."""

    profile = forms.JSONField(required=False, help_text="JSON profile data (will be processed with datason)")

    class Meta:
        model = User
        fields = ["email", "name", "profile", "is_active"]
        widgets = {
            "profile": forms.Textarea(attrs={"rows": 4}),
        }

    def clean_profile(self):
        """Clean profile field with datason processing."""
        profile_data = self.cleaned_data.get("profile")
        if not profile_data:
            return {}

        # Process with datason for consistency
        return datason.serialize(profile_data, config=self.get_datason_config())


# =============================================================================
# URL PATTERNS
# =============================================================================


urlpatterns = [
    # Django API views
    path("api/users/", UserListAPIView.as_view(), name="user-list"),
    path("api/users/<uuid:user_id>/", UserDetailAPIView.as_view(), name="user-detail"),
    path("api/organizations/<uuid:org_id>/", OrganizationAPIView.as_view(), name="organization-detail"),
]

if HAS_DRF:
    from rest_framework.routers import DefaultRouter

    router = DefaultRouter()
    router.register(r"drf/users", UserDRFViewSet)

    urlpatterns += [
        path("api/", include(router.urls)),
    ]

# =============================================================================
# MANAGEMENT COMMANDS
# =============================================================================


class Command(BaseCommand):
    """Django management command for datason operations."""

    help = "Datason integration utilities for Django"

    def add_arguments(self, parser):
        parser.add_argument("--export-users", action="store_true", help="Export all users with datason processing")
        parser.add_argument("--test-config", action="store_true", help="Test datason configuration")

    def handle(self, *args, **options):
        if options["export_users"]:
            self.export_users()
        elif options["test_config"]:
            self.test_config()

    def export_users(self):
        """Export all users with datason processing."""
        users = User.objects.all()
        user_data = [user.to_datason() for user in users]

        processed_data = datason.serialize(user_data, config=DJANGO_API_CONFIG)

        self.stdout.write(self.style.SUCCESS(f"Exported {len(processed_data)} users with datason processing"))
        self.stdout.write(json.dumps(processed_data, indent=2))

    def test_config(self):
        """Test datason configuration."""
        test_data = {
            "user_id": "12345678-1234-5678-9012-123456789abc",
            "created_at": "2023-01-01T12:00:00Z",
            "profile": {"theme": "dark", "settings": {"notifications": True}},
        }

        processed = datason.auto_deserialize(test_data, config=DJANGO_API_CONFIG)

        self.stdout.write(self.style.SUCCESS("Datason configuration test successful"))
        self.stdout.write(f"UUID type: {type(processed['user_id'])}")
        self.stdout.write(f"Datetime type: {type(processed['created_at'])}")


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================


def example_usage():
    """
    Example usage of Django + Datason integration.

    Run this to see the integration in action.
    """

    print("🎯 Django + Datason Integration Example")
    print("=" * 50)

    # Create test user with datason processing
    user_data = {
        "email": "test@example.com",
        "name": "Test User",
        "profile": {"theme": "dark", "language": "en", "preferences": {"notifications": True, "newsletter": False}},
    }

    # Create user using datason processing
    user = User.from_datason(user_data)
    user.save()

    print(f"✅ Created user: {user}")
    print(f"📋 User ID type: {type(user.id)} = {user.id}")

    # Serialize user with datason
    serialized = user.to_datason()
    print(f"📤 Serialized data: {json.dumps(serialized, indent=2)}")

    # Test complex nested data
    complex_data = {
        "user_id": str(user.id),
        "session": {"id": "12345678-1234-5678-9012-123456789abc", "created_at": "2023-01-01T12:00:00Z"},
        "metadata": {"device_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "location": "New York"},
    }

    processed = datason.auto_deserialize(complex_data, config=DJANGO_API_CONFIG)
    print("🔄 Processed complex data:")
    print(f"   User ID type: {type(processed['user_id'])}")
    print(f"   Session ID type: {type(processed['session']['id'])}")
    print(f"   Device ID type: {type(processed['metadata']['device_id'])}")

    print("\n🎉 Django + Datason integration working perfectly!")


if __name__ == "__main__":
    # Setup Django if running standalone
    import django

    django.setup()

    # Run example
    example_usage()
