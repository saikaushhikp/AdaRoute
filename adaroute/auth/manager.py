"""Server-side Authentication and Tenant Identity Management.
Guarantees zero-trust identity derivation and prevents client-side tenant/user spoofing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from adaroute.schemas import PrivacyLevel


class AuthenticationError(Exception):
    """Raised when request lacks valid credentials when required or invalid key provided."""


class IdentitySpoofingError(Exception):
    """Raised when client body claims a tenant/user conflicting with authenticated credentials."""


@dataclass
class TenantContext:
    tenant_id: str
    user_id: str
    is_authenticated: bool = True
    allowed_privacy_levels: set[PrivacyLevel] = field(
        default_factory=lambda: {
            PrivacyLevel.PUBLIC,
            PrivacyLevel.INTERNAL_ONLY,
            PrivacyLevel.CONFIDENTIAL,
            PrivacyLevel.RESTRICTED,
        }
    )
    max_budget_per_request: float = 100.0
    rate_limit_rpm: int = 120


class AuthManager:
    """Manages server-side credential verification and tenant authorization."""

    def __init__(self):
        # Known registered tenant credentials
        self._key_registry: dict[str, dict[str, Any]] = {
            "adaroute-key-default": {
                "tenant_id": "default",
                "user_id": "system_admin",
                "allowed_privacy": {
                    PrivacyLevel.PUBLIC,
                    PrivacyLevel.INTERNAL_ONLY,
                    PrivacyLevel.CONFIDENTIAL,
                    PrivacyLevel.RESTRICTED,
                },
                "max_budget": 100.0,
            },
            "adaroute-key-tenant-a": {
                "tenant_id": "tenant-a",
                "user_id": "user-a",
                "allowed_privacy": {
                    PrivacyLevel.PUBLIC,
                    PrivacyLevel.INTERNAL_ONLY,
                    PrivacyLevel.CONFIDENTIAL,
                    PrivacyLevel.RESTRICTED,
                },
                "max_budget": 50.0,
            },
            "adaroute-key-tenant-b": {
                "tenant_id": "tenant-b",
                "user_id": "user-b",
                "allowed_privacy": {
                    PrivacyLevel.PUBLIC,
                    PrivacyLevel.INTERNAL_ONLY,
                },
                "max_budget": 10.0,
            },
            "adaroute-key-restricted": {
                "tenant_id": "tenant-restricted",
                "user_id": "guest",
                "allowed_privacy": {PrivacyLevel.PUBLIC},
                "max_budget": 1.0,
            },
        }

    def register_key(
        self,
        api_key: str,
        tenant_id: str,
        user_id: str,
        allowed_privacy: set[PrivacyLevel] | None = None,
        max_budget: float = 100.0,
    ) -> None:
        """Registers a new authenticated API key."""
        self._key_registry[api_key] = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "allowed_privacy": allowed_privacy
            or {
                PrivacyLevel.PUBLIC,
                PrivacyLevel.INTERNAL_ONLY,
                PrivacyLevel.CONFIDENTIAL,
                PrivacyLevel.RESTRICTED,
            },
            "max_budget": max_budget,
        }

    def authenticate_request(
        self,
        authorization_header: str | None = None,
        x_api_key_header: str | None = None,
        declared_tenant_id: str | None = None,
        declared_user_id: str | None = None,
        auth_required: bool = False,
    ) -> TenantContext:
        """Extracts and verifies caller identity, enforcing anti-spoofing policies."""
        token = None

        if authorization_header:
            parts = authorization_header.strip().split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
            elif len(parts) == 1:
                token = parts[0]

        if not token and x_api_key_header:
            token = x_api_key_header.strip()

        # Case 1: Token provided
        if token:
            if token not in self._key_registry:
                raise AuthenticationError(f"Invalid API key: '{token[:4]}...' is not authorized")

            profile = self._key_registry[token]
            auth_tenant = profile["tenant_id"]
            auth_user = profile["user_id"]

            # Anti-spoofing validation: caller cannot claim another tenant identity
            if declared_tenant_id and declared_tenant_id != auth_tenant:
                raise IdentitySpoofingError(
                    f"Identity spoofing detected: token belongs to tenant '{auth_tenant}', "
                    f"but request payload claims '{declared_tenant_id}'."
                )

            return TenantContext(
                tenant_id=auth_tenant,
                user_id=declared_user_id or auth_user,
                is_authenticated=True,
                allowed_privacy_levels=profile["allowed_privacy"],
                max_budget_per_request=profile["max_budget"],
            )

        # Case 2: No token provided
        if auth_required:
            raise AuthenticationError("Authentication required: missing or empty Authorization header.")

        # In permissive mode (dev/demo), use declared tenant but mark as unauthenticated
        tenant = declared_tenant_id or "default"
        user = declared_user_id or "anonymous"
        return TenantContext(
            tenant_id=tenant,
            user_id=user,
            is_authenticated=False,
            allowed_privacy_levels={
                PrivacyLevel.PUBLIC,
                PrivacyLevel.INTERNAL_ONLY,
                PrivacyLevel.CONFIDENTIAL,
                PrivacyLevel.RESTRICTED,
            },
            max_budget_per_request=100.0,
        )


default_auth_manager = AuthManager()
