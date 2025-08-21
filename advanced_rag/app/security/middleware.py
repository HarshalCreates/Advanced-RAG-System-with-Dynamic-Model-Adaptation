from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        # Simple query sanitization
        if request.method in {"POST", "PUT", "PATCH"} and request.headers.get("content-type", "").startswith(
            "application/json"
        ):
            # Could implement PII redaction, SQLi patterns, etc.
            pass
        response: Response = await call_next(request)
        # Add basic security headers
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response


